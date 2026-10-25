"""Tiny request/response protocol between a trainer and a scorer subprocess running in its own venv on the same GPU.

The scorer venvs need library versions qwen-tts cannot share (Qwen3-ASR wants transformers 5, qwen-tts pins 4.57), so scoring runs
as `<venv python> -m <module>`. One request id per stdin line; request = pickle /tmp/ipc/req_<id>.pkl, response = pickle
/tmp/ipc/res_<id>.pkl, then "DONE <id>" on stdout. The server logs to stderr. No heavy imports here.
"""
import os
import pickle
import subprocess
import sys
import threading
import traceback

IPC = "/tmp/ipc"


class Client:
    def __init__(self, python, module, env=None):
        self.python, self.module, self.env = python, module, env or {}
        self.n, self.fut, self.lock = 0, {}, threading.Lock()
        self.start()

    def start(self):
        os.makedirs(IPC, exist_ok=True)
        env = {**os.environ, "PYTHONPATH": "/root", "HF_HOME": "/opt/hf", "TOKENIZERS_PARALLELISM": "false", **self.env}
        self.p = subprocess.Popen([self.python, "-u", "-m", self.module], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                  env=env, cwd="/root")
        while True:  # libraries may print to stdout while loading; the protocol starts at READY
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f"{self.module} exited before READY")
            if line.startswith("READY"):
                break
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            if line.startswith("DONE "):
                with self.lock:
                    ev = self.fut.get(line.split()[1])
                if ev:
                    ev.set()
        with self.lock:  # server died: wake every waiter
            for ev in self.fut.values():
                ev.set()

    def submit(self, payload):
        self.n += 1
        rid = f"{self.module.rsplit('.', 1)[-1]}_{self.n}"
        with open(f"{IPC}/req_{rid}.pkl.tmp", "wb") as f:
            pickle.dump(payload, f)
        os.replace(f"{IPC}/req_{rid}.pkl.tmp", f"{IPC}/req_{rid}.pkl")
        ev = threading.Event()
        with self.lock:
            self.fut[rid] = ev
        self.p.stdin.write(rid + "\n")
        self.p.stdin.flush()
        return rid

    def result(self, rid, timeout=3600):
        with self.lock:
            ev = self.fut[rid]
        ev.wait(timeout)
        p = f"{IPC}/res_{rid}.pkl"
        if not os.path.exists(p):
            raise RuntimeError(f"no result for {rid} (alive={self.alive()})")
        r = pickle.load(open(p, "rb"))
        os.remove(p)
        with self.lock:
            self.fut.pop(rid, None)
        if "error" in r:
            raise RuntimeError(f"{self.module} error: {r['error']}")
        return r["ok"]

    def __call__(self, payload):
        return self.result(self.submit(payload))

    def alive(self):
        return self.p.poll() is None

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=120)
        except Exception:  # noqa: BLE001
            self.p.kill()


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def serve(handler):
    """Run a server loop: handler(payload) -> result. Call after the models are loaded."""
    os.makedirs(IPC, exist_ok=True)
    print("READY", flush=True)
    for line in sys.stdin:
        rid = line.strip()
        if not rid:
            continue
        try:
            req = pickle.load(open(f"{IPC}/req_{rid}.pkl", "rb"))
            os.remove(f"{IPC}/req_{rid}.pkl")
            res = {"ok": handler(req)}
        except Exception as e:  # noqa: BLE001
            log("[server] ERROR", traceback.format_exc())
            res = {"error": repr(e)[:500]}
        with open(f"{IPC}/res_{rid}.pkl.tmp", "wb") as f:
            pickle.dump(res, f)
        os.replace(f"{IPC}/res_{rid}.pkl.tmp", f"{IPC}/res_{rid}.pkl")
        print(f"DONE {rid}", flush=True)
