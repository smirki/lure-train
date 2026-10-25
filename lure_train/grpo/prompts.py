"""Prompts for the robustness (R) branch. Local, no models, $0.

  python3 -m lure_train.grpo.prompts --docs my_docs.jsonl --out inputs/
  modal volume put -f "$LURE_VOLUME" inputs/hard.jsonl inputs/long.jsonl /inputs/

hard.jsonl  numbers, dates, money, units, abbreviations, acronyms, lists, run-on sentences (template-built; each item carries the
            written text plus spoken renderings `alts`; WER takes the min over the written form and the alts)
long.jsonl  30-75 s pieces: consecutive paragraphs of your own documents (80-180 plain words), tagged or plain, sometimes with the
            first paragraph's direction. --docs rows: {"paragraphs": [{"text" (inline tags), "plain", "direction"?}], "genre"?}
Row schema: {id, kind, text, plain, alts, direction, est_s, n_tags}; est_s = 2.6 words/s + 0.7 s per tag.
"""
import argparse
import hashlib
import json
import os
import random
import re

TAG = re.compile(r"\(([^()]{1,40})\)")
WPS = 2.6
rng = random.Random(7)


def est_s(plain, text):
    return round(len(plain.split()) / WPS + 0.7 * len(TAG.findall(text)), 1)


def hid(s):
    return hashlib.sha1(s.encode()).hexdigest()[:12]


NAMES = ["Patel", "Okafor", "Lindqvist", "Nakamura", "Alvarez", "Brennan", "Kowalski", "Haddad", "Moreau", "Osei", "Fitzgerald", "Tanaka"]
FIRST = ["Priya", "Marcus", "Ingrid", "Kenji", "Lucia", "Declan", "Amara", "Tomasz", "Yusuf", "Celine", "Rosa", "Theo"]
ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
MON_ABBR = ["Jan.", "Feb.", "Mar.", "Apr.", "May", "Jun.", "Jul.", "Aug.", "Sept.", "Oct.", "Nov.", "Dec."]


def w2(n):
    """0-99 in words."""
    if n < 10:
        return ONES[n]
    if n < 20:
        return TEENS[n - 10]
    return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])


def w3(n):
    if n < 100:
        return w2(n)
    h, r = divmod(n, 100)
    return ONES[h] + " hundred" + ("" if r == 0 else " " + w2(r))


def wint(n):
    if n < 1000:
        return w3(n)
    if n < 1_000_000:
        t, r = divmod(n, 1000)
        return w3(t) + " thousand" + ("" if r == 0 else " " + w3(r))
    m, r = divmod(n, 1_000_000)
    return w3(m) + " million" + ("" if r == 0 else " " + wint(r))


def ordw(n):
    w = w2(n).split()
    last = w[-1]
    w[-1] = ORD.get(last, last[:-1] + "ieth" if last.endswith("y") else last + "th")
    return " ".join(w)


def year(y):
    a, b = divmod(y, 100)
    if b == 0:
        return w2(a) + " hundred"
    if a == 20 and b < 10:
        return "two thousand " + ONES[b]
    return w2(a) + " " + (("oh " + ONES[b]) if b < 10 else w2(b))


# each slot -> (written, [spoken alternates])
def s_money():
    d, c = rng.randint(1, 499), rng.choice([0, 5, 25, 49, 50, 75, 99])
    if c == 0:
        return f"${d}", [f"{wint(d)} dollars"]
    return f"${d}.{c:02d}", [f"{wint(d)} {w2(c)}", f"{wint(d)} dollars and {w2(c)} cents"]


def s_time():
    h, m = rng.randint(1, 12), rng.choice([0, 5, 15, 30, 45, 50, 7])
    ap = rng.choice(["a.m.", "p.m."])
    sp = "a m" if ap == "a.m." else "p m"
    if m == 0:
        return f"{h}:00 {ap}", [f"{w2(h)} {sp}", f"{w2(h)} o'clock {sp}"]
    mm = ("oh " + ONES[m]) if m < 10 else w2(m)
    return f"{h}:{m:02d} {ap}", [f"{w2(h)} {mm} {sp}"]


def s_date():
    mo, d, y = rng.randrange(12), rng.randint(1, 28), rng.randint(1962, 2031)
    if rng.random() < 0.5:
        sfx = {1: "st", 2: "nd", 3: "rd", 21: "st", 22: "nd", 23: "rd"}.get(d, "th")
        return f"{MONTHS[mo]} {d}{sfx}, {y}", [f"{MONTHS[mo].lower()} {ordw(d)} {year(y)}"]
    return f"{MON_ABBR[mo]} {d}, {y}", [f"{MONTHS[mo].lower()} {ordw(d)} {year(y)}", f"{MONTHS[mo].lower()} {w2(d)} {year(y)}"]


def s_pct():
    a, b = rng.randint(1, 99), rng.choice([None, 5, 2, 75])
    if b is None:
        return f"{a}%", [f"{w2(a)} percent"]
    return f"{a}.{b}%", [f"{w2(a)} point {' '.join(ONES[int(ch)] for ch in str(b))} percent"]


def s_big():
    n = rng.choice([rng.randint(1100, 9999), rng.randint(10_000, 999_999), rng.randint(1, 9) * 1_000_000 + rng.randint(1, 999) * 1000])
    alts = [wint(n)]
    if 1100 <= n < 10000 and n % 100:
        alts.append(w2(n // 100) + " hundred " + w2(n % 100))
    return f"{n:,}", alts


def s_phone():
    a = "".join(str(rng.randint(0, 9)) for _ in range(3))
    b = "".join(str(rng.randint(0, 9)) for _ in range(4))
    digits = "555" + a + b
    return f"(555) {a}-{b}", [" ".join(ONES[int(c)] for c in digits), " ".join(("oh" if c == "0" else ONES[int(c)]) for c in digits)]


def s_unit():
    n = rng.randint(2, 240)
    u = rng.choice([("mph", "miles per hour"), ("km", "kilometers"), ("kg", "kilograms"), ("mg", "milligrams"), ("ml", "milliliters"),
                    ("°F", "degrees fahrenheit"), ("°C", "degrees celsius"), ("ft", "feet"), ("lbs", "pounds"), ("GB", "gigabytes")])
    sep = "" if u[0].startswith("°") else " "
    return f"{n}{sep}{u[0]}", [f"{wint(n)} {u[1]}"]


def s_dec():
    a, b = rng.randint(0, 99), rng.randint(1, 99)
    return f"{a}.{b}", [f"{w2(a)} point {' '.join(ONES[int(c)] for c in str(b))}"]


def s_ord():
    n = rng.randint(2, 99)
    sfx = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{sfx}", [ordw(n)]


def s_dr():
    t, sp = rng.choice([("Dr.", "doctor"), ("Mr.", "mister"), ("Mrs.", "missus"), ("Ms.", "miz"), ("Prof.", "professor"), ("Sgt.", "sergeant"),
                        ("Capt.", "captain"), ("Gov.", "governor")])
    n = rng.choice(NAMES)
    return f"{t} {n}", [f"{sp} {n.lower()}"]


def s_addr():
    n, st = rng.randint(12, 4999), rng.choice(["Elm", "Maple", "Harbor", "Juniper", "Lake", "Fifth"])
    k, sp = rng.choice([("St.", "street"), ("Ave.", "avenue"), ("Blvd.", "boulevard"), ("Rd.", "road"), ("Apt. 4B", "apartment four b")])
    if k.startswith("Apt"):
        return f"{n} {st} Ln., {k}", [f"{wint(n)} {st.lower()} lane {sp}"]
    return f"{n} {st} {k}", [f"{wint(n)} {st.lower()} {sp}"]


def s_acr():
    a, sp = rng.choice([("FBI", "f b i"), ("NASA", "nasa"), ("CEO", "c e o"), ("ASAP", "a s a p"), ("HVAC", "h vac"), ("GPS", "g p s"),
                        ("ATM", "a t m"), ("FAQ", "f a q"), ("UNESCO", "unesco"), ("PDF", "p d f"), ("DIY", "d i y"), ("RSVP", "r s v p"),
                        ("NATO", "nato"), ("IRS", "i r s"), ("MRI", "m r i"), ("U.S.", "u s"), ("U.K.", "u k")])
    return a, [sp]


def s_latin():
    a, sp = rng.choice([("e.g.", ["for example", "e g"]), ("i.e.", ["that is", "i e"]), ("etc.", ["et cetera"]), ("vs.", ["versus", "vs"]),
                        ("approx.", ["approximately"]), ("No. 7", ["number seven"]), ("Inc.", ["incorporated", "inc"]), ("Ltd.", ["limited"]),
                        ("Jr.", ["junior"]), ("Mt. Rainier", ["mount rainier"])])
    return a, sp


SLOTS = {"money": s_money, "time": s_time, "date": s_date, "pct": s_pct, "big": s_big, "phone": s_phone, "unit": s_unit, "dec": s_dec,
         "ord": s_ord, "dr": s_dr, "addr": s_addr, "acr": s_acr, "latin": s_latin}
TEMPLATES = [
    "{dr} said the clinic opens at {time} on {date}, and the copay is still {money}.",
    "Your order of {big} units ships on {date} from {addr}; call {phone} if it's late.",
    "The {acr} report says prices rose {pct} last quarter, roughly {money} per household.",
    "We left at {time}, drove {unit} in the rain, and reached {addr} a little after midnight.",
    "Please RSVP by {date}. The dinner is {money} a plate, and parking at {addr} is free.",
    "The {ord} runner finished in {dec} seconds, just ahead of {dr}, who came in {ord}.",
    "{dr} measured {unit} on Tuesday and {unit} on Wednesday, which is {pct} higher.",
    "Revenue hit {big} dollars, up {pct}, and the {acr} wants a summary by {time} tomorrow.",
    "Bring the basics, {latin} water, a flashlight, and about {money} in cash, to {addr}.",
    "The meeting moved from {time} to {time}, so tell {dr} and the team at {acr} {latin}",
    "On {date} the population was {big}; by {date} it had grown {pct}.",
    "Flight {ord} on the list departs at {time}; the fare went from {money} to {money} overnight.",
    "Call {phone} or write to {dr} at {addr} before {date}.",
    "The recipe needs {unit} of flour, {dec} cups of sugar, and an oven at {unit}.",
    "Our {ord} anniversary is {date}, and dinner for two is {money} before tip.",
    "The {acr} fined the company {money} per day, {latin} about {big} dollars a year.",
]
LIST_ITEMS = ["two dozen eggs", "a bag of flour", "three lemons", "olive oil", "paper towels", "a roll of tape", "batteries", "dish soap",
              "a jar of honey", "frozen peas", "the blue folder", "my passport", "spare keys", "a phone charger", "sunscreen", "rain boots",
              "a notebook", "the extension cord", "dog food", "coffee filters", "a wrench", "string lights", "birthday candles", "cough drops"]
CLAUSES = ["I got to the station late", "the train had already left", "so I called my sister", "she didn't pick up",
           "which honestly was typical", "then it started raining", "and my umbrella was in the car", "the car was at the shop",
           "the shop closes at five", "nobody told me that", "a stranger offered to share a cab", "we ended up talking the whole ride",
           "it turns out he grew up two streets from me", "his mother taught my mother piano", "by the time we got downtown it was dark",
           "the restaurant had given away our table", "the manager felt bad about it", "she found us a booth in the back",
           "the food came out cold", "we laughed about it anyway", "my phone died halfway through dessert", "I had to borrow his charger",
           "he missed his own train because of me", "we still text every week"]
JOIN = [", and", ", so", ", but", " and then", ", which meant", " because", ", and somehow", " while"]


def fill(tpl):
    written, alts_a, alts_b = tpl, tpl, tpl
    for m in re.finditer(r"\{(\w+)\}", tpl):
        w, alts = SLOTS[m.group(1)]()
        written = written.replace(m.group(0), w, 1)
        alts_a = alts_a.replace(m.group(0), alts[0], 1)
        alts_b = alts_b.replace(m.group(0), alts[-1], 1)
    return written, sorted({alts_a, alts_b})


def hard_items(n_tpl=8, n_list=40, n_run=50):
    out = []
    for tpl in TEMPLATES:
        for _ in range(n_tpl):
            w, alts = fill(tpl)
            out.append({"sub": "numbers_abbr", "text": w, "plain": w, "alts": alts})
    for _ in range(n_list):
        k = rng.randint(5, 9)
        items = rng.sample(LIST_ITEMS, k)
        lead = rng.choice(["Before we go, grab", "On the list for Saturday:", "Don't forget", "The box should have", "Okay, so we still need"])
        w = f"{lead} {', '.join(items[:-1])}, and {items[-1]}."
        out.append({"sub": "list", "text": w, "plain": w, "alts": []})
    for _ in range(n_run):
        k = rng.randint(6, 10)
        cl = rng.sample(CLAUSES, k)
        s = cl[0]
        for c in cl[1:]:
            j = rng.choice(JOIN)
            s += (", " if c.startswith(("so ", "and ", "which ", "then ")) else j + " ") + c
        w = s[0].upper() + s[1:] + "."
        out.append({"sub": "run_on", "text": w, "plain": w, "alts": []})
    rows = []
    for r in out:
        r.update(id="hd_" + hid(r["text"]), kind="hard", direction=None, est_s=est_s(r["plain"], r["text"]), n_tags=0)
        rows.append(r)
    return rows


# ------------------------------------------------------------------------------------------------ long-form
def long_items(docs_path, n=700, wmin=80, wmax=180):
    docs = [json.loads(line) for line in open(docs_path, encoding="utf-8") if line.strip()]
    rng.shuffle(docs)
    rows = []
    for d in docs:
        ps = d["paragraphs"]
        if len(ps) < 2:
            continue
        a = rng.randrange(len(ps))
        sel, wc = [], 0
        for p in ps[a:] + ps[:a]:
            if wc >= wmin:
                break
            sel.append(p)
            wc += len(p["plain"].split())
        if not wmin <= wc <= wmax:
            continue
        tagged = rng.random() < 0.5
        text = " ".join(p["text"] if tagged else p["plain"] for p in sel)
        plain = " ".join(p["plain"] for p in sel)
        direction = sel[0].get("direction") if rng.random() < 0.3 else None
        rows.append({"id": "lg_" + hid(text), "kind": "long", "sub": d.get("genre"), "text": text, "plain": plain, "alts": [],
                     "direction": direction, "est_s": est_s(plain, text), "n_tags": len(TAG.findall(text))})
        if len(rows) >= n:
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="", help="jsonl of documents for long.jsonl (skipped if empty)")
    ap.add_argument("--out", default="inputs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sets = {"hard": hard_items()}
    if a.docs:
        sets["long"] = long_items(a.docs)
    for name, rows in sets.items():
        with open(os.path.join(a.out, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        print(name, len(rows))


if __name__ == "__main__":
    main()
