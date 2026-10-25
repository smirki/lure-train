"""Grouped fast code-predictor (CP) heads. V2s is the Lure Turbo head.

The stock CP (5 layers, hidden 1024) runs 15 sequential forwards per 80 ms frame: position k (1..15) sees emb(cb_{k-1}) and predicts
cb_k with lm_head[k-1]; position 0 is the talker hidden state, position 1 the talker embedding of cb0. A FastCP head keeps that
architecture (initialised from the stock CP) and splits positions 1..15 into groups. One forward per group sees the real code
embeddings up to the group's first position and a learned mask vector after it, so a group-first position is exactly the stock
computation (causal attention).

  V2   groups 1-3, 4-7, 8-15: 3 passes, codebooks inside a group predicted independently (warm-up trunk for V2s)
  V2s  same 3 passes, but inside a group codebooks are sampled one by one: the trunk output at a position is corrected by a small
       MLP (hidden 512) fed with the sum of the embeddings already sampled in the group and the previous code's embedding. Costs a
       few small matmuls per codebook, not a transformer pass. Trunk from V2, add-on from a zero-output init.
Generation patches talker.code_predictor.generate (CPPatch), so the rest of qwen-tts generation is unchanged.
"""
from ..common import NCB, vpath

HEADS = {"V2": [(1, 3), (4, 7), (8, 15)], "V2s": [(1, 3), (4, 7), (8, 15)]}
ADDON = {"V2s"}


def make_fastcp(cp, emb0, groups, addon=False):
    import copy
    import torch
    from transformers.masking_utils import create_causal_mask

    class FastCP(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.groups = [tuple(g) for g in groups]
            s.layers = copy.deepcopy(cp.model.layers)
            s.norm = copy.deepcopy(cp.model.norm)
            s.rotary = copy.deepcopy(cp.model.rotary_emb)
            s.lm_head = copy.deepcopy(cp.lm_head)
            s.proj = copy.deepcopy(cp.small_to_mtp_projection)
            s.mask = torch.nn.Parameter(torch.stack([cp.model.codec_embedding[k - 2].weight.detach().float().mean(0) for k in range(2, 16)]))
            s._ext = (cp.model.codec_embedding, emb0)  # frozen, shared with the talker (tuple: not registered as submodules)
            s.cfg = cp.model.config
            s._W, s._mc = {}, {}  # inference caches: stacked lm_head weights per group, causal mask / rotary per length
            s.addon = None
            if addon:  # z_k += W2 silu(W1 [ln(z_k); ln(sum of in-group picked code embeddings); ln(emb of cb_{k-1})] + pos_k), W2 zero-init
                Hd = cp.model.config.hidden_size
                s.addon = torch.nn.ModuleDict({"n1": torch.nn.LayerNorm(Hd), "n2": torch.nn.LayerNorm(Hd), "n3": torch.nn.LayerNorm(Hd),
                                               "w1": torch.nn.Linear(3 * Hd, 512), "w2": torch.nn.Linear(512, Hd)})
                s.addon_pos = torch.nn.Parameter(torch.zeros(NCB - 1, 512))
                torch.nn.init.zeros_(s.addon["w2"].weight)
                torch.nn.init.zeros_(s.addon["w2"].bias)

        def add(s, z, csum, prev, k):
            a = s.addon
            u = a["w1"](torch.cat([a["n1"](z), a["n2"](csum), a["n3"](prev)], -1)) + s.addon_pos[k - 1].to(z.dtype)
            return z + a["w2"](torch.nn.functional.silu(u))

        def run(s, X):
            z = s.proj(X)
            L = z.shape[1]
            key = (L, z.shape[0], z.dtype)
            if not s.training and key in s._mc:
                cp2, p2, mask, pe = s._mc[key]
            else:
                cp2 = torch.arange(L, device=z.device)
                p2 = cp2.unsqueeze(0)
                mask = create_causal_mask(config=s.cfg, input_embeds=z, attention_mask=None, cache_position=cp2, past_key_values=None, position_ids=p2)
                pe = s.rotary(z, p2)
                if not s.training:
                    s._mc[key] = (cp2, p2, mask, pe)
            for lyr in s.layers:
                z = lyr(z, attention_mask=mask, position_ids=p2, past_key_values=None, use_cache=False, cache_position=cp2, position_embeddings=pe)[0]
            return s.norm(z)

        def real_embeds(s, h, codes):
            """h (N, H) talker hidden, codes (N, 16) -> (N, 16, H) = [h, emb0(cb0), cemb[0](cb1), ..., cemb[13](cb14)]."""
            cemb, e0 = s._ext
            return torch.cat([h.unsqueeze(1), e0(codes[:, :1]).to(h.dtype)] + [cemb[i - 1](codes[:, i:i + 1]).to(h.dtype) for i in range(1, NCB - 1)], 1)

        def group_logits(s, E, g):
            """Teacher-forced logits of group g = (first, last): (N, last - first + 1, 2048)."""
            first, last = g
            X = E[:, :last + 1].clone()
            if last > first:
                X[:, first + 1:last + 1] = s.mask[first - 1:last - 1].to(X.dtype)
            Z = s.run(X)
            if s.addon is not None and last > first:  # in-group context from the real codes
                ks = torch.arange(first + 1, last + 1, device=X.device)
                prevs = E[:, first + 1:last + 1]
                Z = torch.cat([Z[:, :first + 1], s.add(Z[:, first + 1:last + 1], prevs.cumsum(1), prevs, ks)], 1)
            return torch.stack([s.lm_head[k - 1](Z[:, k]) for k in range(first, last + 1)], 1)

        def logp(s, h, codes):
            """Teacher-forced log-probs of cb1-15: (N, 15, 2048)."""
            E = s.real_embeds(h, codes)
            return torch.cat([s.group_logits(E, g) for g in s.groups], 1).float().log_softmax(-1)

        @torch.no_grad()
        def sample_codes(s, x2, temp=0.9, top_k=50):
            """x2 (B, 2, H) = [h, emb0(cb0)] -> cb1-15 (B, 15)."""
            cemb, _ = s._ext
            B = x2.shape[0]
            out = torch.zeros(B, NCB - 1, dtype=torch.long, device=x2.device)
            X = x2
            for first, last in s.groups:
                X_ = torch.cat([X, s.mask[first - 1:last - 1].to(X.dtype).unsqueeze(0).expand(B, -1, -1)], 1) if last > first else X
                Z = s.run(X_)
                if s.addon is not None and last > first:  # in-group codebooks one by one through the add-on
                    out[:, first - 1] = sample(s.lm_head[first - 1](Z[:, first]), temp, top_k)
                    csum = None
                    for k in range(first + 1, last + 1):
                        prev = cemb[k - 2](out[:, k - 2]).to(Z.dtype)
                        csum = prev if csum is None else csum + prev
                        out[:, k - 1] = sample(s.lm_head[k - 1](s.add(Z[:, k], csum, prev, k)), temp, top_k)
                else:  # all heads of the group in one batched matmul
                    if (first, last) not in s._W:
                        s._W[(first, last)] = torch.stack([s.lm_head[k - 1].weight for k in range(first, last + 1)])
                    out[:, first - 1:last] = sample(torch.einsum("bnh,nvh->bnv", Z[:, first:last + 1], s._W[(first, last)]), temp, top_k)
                X = torch.cat([X] + [cemb[k - 1](out[:, k - 1:k]) for k in range(first, min(last, NCB - 2) + 1)], 1)
            return out

    return FastCP()


def sample(logits, temp=0.9, top_k=50):
    """Top-k + temperature sampling over the last dim (Gumbel max)."""
    import torch
    lg = logits.float() / temp
    v = lg.topk(top_k, -1).values[..., -1:]
    lg = lg.masked_fill(lg < v, float("-inf"))
    return (lg - torch.log(-torch.log(torch.rand_like(lg).clamp_(1e-10, 1.0)))).argmax(-1)


class CPPatch:
    """Swap talker.code_predictor.generate for a FastCP head (head=None restores the stock CP)."""

    def __init__(self, st):
        self.cp = st.m.talker.code_predictor
        self.orig = self.cp.generate

    def set(self, head=None, temp=0.9):
        import types
        if head is None:
            self.cp.generate = self.orig
            return

        def gen(inputs_embeds=None, **kw):
            return types.SimpleNamespace(sequences=head.sample_codes(inputs_embeds, temp))
        self.cp.generate = gen


def new_head(st, name):
    tal = st.m.talker
    return make_fastcp(tal.code_predictor, tal.get_input_embeddings(), HEADS[name], addon=name in ADDON)


def load_head(st, name, path, strict=True):
    """bf16 eval head from a saved state dict (strict=False: V2s add-on missing from a V2 trunk stays at its zero init)."""
    import torch
    h = new_head(st, name)
    h.load_state_dict(torch.load(vpath(path), map_location="cuda:0"), strict=strict)
    h._W = {}
    return h.to(device="cuda:0", dtype=torch.bfloat16).eval()
