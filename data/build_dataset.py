"""
Cleans up phrasings/raw.jsonl and turns them into training dataset.

Follows the "Build script guide" tab of the project doc, steps 1-9:
  1 filter  2 hand-review rejects  3 split  4 fill slots  5 sequences
  6 refusals  7 noise  8 balance/check/write  9 human test set check

Run from the repo root:
    python data/build_dataset.py
    python data/build_dataset.py --train-size 40000 --test-size 4000
Outputs go to data/out/: train.jsonl, test.jsonl, stats.txt, split.json,
output_vocab.txt. Review samples go to data/phrasings/review/.
"""

import argparse
import json
import math
import random
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path

random.seed(42)

HERE = Path(__file__).resolve().parent
RAW_PATH = HERE / "phrasings" / "raw.jsonl"
REJECTS_PATH = HERE / "phrasings" / "rejects.txt"
REVIEW_DIR = HERE / "phrasings" / "review"
OUT_DIR = HERE / "out"
HUMAN_TEST_PATH = HERE / "human_test.jsonl"

MAX_TOKENS = 128          # model context: input chars + output tokens + specials
SPECIAL_TOKENS = 2

# ---------------------------------------------------------------------------
# Rules as data
# ---------------------------------------------------------------------------

SLOTS = {
    "FWD_DIST": ["DIST"], "FWD_TIME": ["TIME"], "FWD_PLAIN": [], "FWD_SMALL": [],
    "BACK_DIST": ["DIST"], "BACK_TIME": ["TIME"], "BACK_PLAIN": [], "BACK_SMALL": [],
    "TURN_DIR": ["DIR"], "TURN_DIR_ANGLE": ["ANGLE", "DIR"], "TURN_DIR_TIME": ["DIR", "TIME"],
    "TURN_DIR_SMALL": ["DIR"], "TURN_PLAIN": [], "TURN_ANGLE": ["ANGLE"], "TURN_TIME": ["TIME"],
    "TURN_AROUND": [],
    "FWD_UNTIL": ["OBJ"], "SEARCH": ["OBJ"], "WATCH_LEAD": ["OBJ"], "WATCH_TRAIL": ["OBJ"],
    "CONNECTOR": [], "STOP": [], "UNKNOWN": [],
}


def fmt_s(t):
    return "%ss" % ("%g" % t)


# Label for each intent, given filled slot values. Change defaults here only.
LABELS = {
    "FWD_DIST":       lambda v: "FWD %d" % v["DIST"],
    "FWD_TIME":       lambda v: "FWD " + fmt_s(v["TIME"]),
    "FWD_PLAIN":      lambda v: "FWD 2s",
    "FWD_SMALL":      lambda v: "FWD 1s",
    "BACK_DIST":      lambda v: "BACK %d" % v["DIST"],
    "BACK_TIME":      lambda v: "BACK " + fmt_s(v["TIME"]),
    "BACK_PLAIN":     lambda v: "BACK 2s",
    "BACK_SMALL":     lambda v: "BACK 1s",
    "TURN_DIR":       lambda v: "%s 90" % v["DIR"],
    "TURN_DIR_ANGLE": lambda v: "%s %d" % (v["DIR"], v["ANGLE"]),
    "TURN_DIR_TIME":  lambda v: "%s %s" % (v["DIR"], fmt_s(v["TIME"])),
    "TURN_DIR_SMALL": lambda v: "%s 30" % v["DIR"],
    "TURN_PLAIN":     lambda v: "LEFT 90",
    "TURN_ANGLE":     lambda v: "LEFT %d" % v["ANGLE"],
    "TURN_TIME":      lambda v: "LEFT " + fmt_s(v["TIME"]),
    "TURN_AROUND":    lambda v: "LEFT 180",
    "FWD_UNTIL":      lambda v: "FWD UNTIL %s" % v["OBJ"],
    "SEARCH":         lambda v: "SEARCH %s" % v["OBJ"],
    "WATCH_LEAD":     lambda v: "WATCH %s" % v["OBJ"],
    "WATCH_TRAIL":    lambda v: "WATCH %s" % v["OBJ"],
    "STOP":           lambda v: "STOP",
    "UNKNOWN":        lambda v: "UNKNOWN",
}

# TURN_PLAIN phrasings that say spin/twirl/whirl get LEFT 360 instead of LEFT 90.
SPIN_RE = re.compile(r"\b(spin|spinning|spins|twirl\w*|whirl\w*)\b", re.I)

MOVE_INTENTS = [i for i in SLOTS if i not in
                ("WATCH_LEAD", "WATCH_TRAIL", "CONNECTOR", "STOP", "UNKNOWN")]
OBJ_INTENTS = ["FWD_UNTIL", "SEARCH"]

NUMBER_WORDS = set("""zero one two three four five six seven eight nine ten eleven twelve
thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty
sixty seventy eighty ninety hundred thousand half quarter couple few dozen twice""".split())

# Step 1 keyword rules: a phrasing of this intent is rejected if it contains any of these.
_FWD = ["back", "backward", "backwards", "reverse", "left", "right", "turn", "until", "stop", "find"]
_BACK = ["forward", "forwards", "ahead", "left", "right", "until", "find", "turn"]
_TURN_DIR = ["forward", "forwards", "backward", "backwards", "until", "find", "around"]
_TURN = ["left", "right", "forward", "forwards", "until", "find", "around"]
_PLAIN = ["little", "bit", "slightly", "far", "tiny"]
KEYWORDS = {
    "FWD_DIST": _FWD, "FWD_TIME": _FWD, "FWD_PLAIN": _FWD + _PLAIN, "FWD_SMALL": _FWD,
    "BACK_DIST": _BACK, "BACK_TIME": _BACK, "BACK_PLAIN": _BACK + _PLAIN, "BACK_SMALL": _BACK,
    "TURN_DIR": _TURN_DIR, "TURN_DIR_ANGLE": _TURN_DIR, "TURN_DIR_TIME": _TURN_DIR,
    "TURN_DIR_SMALL": _TURN_DIR, "TURN_PLAIN": _TURN + _PLAIN, "TURN_ANGLE": _TURN,
    "TURN_TIME": _TURN,
    "TURN_AROUND": ["left", "right", "until", "find", "full"],
    "FWD_UNTIL": ["find", "look", "search", "left", "right", "back"],
    "SEARCH": ["forward", "back", "left", "right"],
    "WATCH_LEAD": ["forward", "back", "left", "right", "turn", "find", "search"],
    "WATCH_TRAIL": ["forward", "back", "left", "right", "turn", "find", "search"],
    "CONNECTOR": ["go", "turn", "stop", "find", "drive", "move"],
    "STOP": ["forward", "back", "left", "right", "turn", "find"],
    "UNKNOWN": ["bottle", "cup", "mug", "chair", "person", "human"],
}

OBJ_SYNONYMS = {
    "bottle": ["bottle", "water bottle", "plastic bottle", "soda bottle"],
    "cup": ["cup", "mug", "coffee cup"],
    "chair": ["chair", "office chair"],
    "person": ["person", "human", "guy"],
}
UNKNOWN_OBJECTS = ["shoe", "hat", "key", "wallet", "sock", "door", "pen", "box"]
NEGATIONS = {"don't", "dont", "not", "never", "no", "do", "doesn't"}

LIGHT_JOINERS = [",", "and", "then", ", then", "and then"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PUNCT = "".join(c for c in string.punctuation if c not in "{}")
_PUNCT_RE = re.compile("[" + re.escape(_PUNCT) + "]")
SLOT_RE = re.compile(r"\{([A-Z]+)\}")


def normalize(text):
    text = _PUNCT_RE.sub(" ", text.lower().replace("’", "'"))
    return " ".join(text.split())


def words_of(text):
    return re.findall(r"[a-z']+", text.lower().replace("’", "'"))


def out_tokens(label):
    return re.findall(r";|[^\s;]+", label)


_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def int_words(n):
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else "-" + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        head = ("a hundred" if random.random() < 0.5 else "one hundred") if n // 100 == 1 else _ONES[n // 100] + " hundred"
        return head + ("" if rest == 0 else " and " + int_words(rest))
    return str(n)


def say_number(x, words_p=0.3):
    """Digits about 70% of the time, words otherwise. Handles .5 values."""
    whole = int(x)
    half = abs(x - whole - 0.5) < 1e-9
    if random.random() >= words_p:
        return ("%g" % x)
    if half:
        return "half a" if whole == 0 else int_words(whole) + " and a half"
    return int_words(whole)


def with_unit(x, singular, plural, abbrevs=()):
    """Number plus unit, with correct singular/plural and some abbreviations."""
    num = say_number(x)
    is_one = abs(x - 1) < 1e-9
    if num == "half a":                       # "half a meter", never "half a meters"
        return "half a " + singular
    if abbrevs and num[0].isdigit() and random.random() < 0.4:
        ab = random.choice(abbrevs)
        return num + ab if random.random() < 0.5 else num + " " + ab
    if "and a half" in num:
        return num + " " + plural
    return num + " " + (singular if is_one else plural)


def round5(x):
    return int(5 * round(x / 5.0))


# ---------------------------------------------------------------------------
# Step 4: slot fillers. Each returns (text, label value).
# ---------------------------------------------------------------------------

def fill_dist():
    r = random.random()
    if r < 0.05:
        m = random.choice([10, 20, 50, 100])
        return with_unit(m, "meter", "meters", ("m",)), m * 100
    if r < 0.17:
        return random.choice([("half a meter", 50), ("a meter and a half", 150), ("a foot", 30),
                              ("a meter", 100), ("a couple of meters", 200), ("a couple of feet", 60)])
    unit = random.choice(["cm", "m", "m", "ft", "in"])
    if unit == "cm":
        n = random.choice(range(10, 301, 5))
        return with_unit(n, "centimeter", "centimeters", ("cm",)), n
    if unit == "m":
        n = random.choice([0.5, 1, 1.5, 2, 2.5, 3, 4, 5])
        return with_unit(n, "meter", "meters", ("m",)), int(n * 100)
    if unit == "ft":
        n = random.choice([1, 2, 3, 4, 5, 6, 8, 10])
        return with_unit(n, "foot", "feet", ("ft",)), round5(n * 30.48)
    n = random.choice([6, 10, 12, 18, 24, 30, 36])
    return with_unit(n, "inch", "inches", ("in",)), round5(n * 2.54)


def fill_time():
    if random.random() < 0.15:
        return random.choice([("half a second", 0.5), ("a second", 1),
                              ("a couple of seconds", 2), ("a few seconds", 3)])
    t = random.choice([0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 10, 12, 15])
    return with_unit(t, "second", "seconds", ("s", "sec", "secs")), t


def fill_angle():
    if random.random() < 0.15:
        return random.choice([("a quarter turn", 90), ("a half turn", 180),
                              ("a full circle", 360), ("a full turn", 360)])
    a = random.choice([15, 30, 45, 60, 90, 90, 120, 135, 180, 180, 270, 360])
    return with_unit(a, "degree", "degrees", ("deg",)), a


def fill_dir():
    d = random.choice(["left", "right"])
    return d, d.upper()


def fill_obj(unknown=False):
    if unknown:
        o = random.choice(UNKNOWN_OBJECTS)
        return o, o
    cls = random.choice(list(OBJ_SYNONYMS))
    return random.choice(OBJ_SYNONYMS[cls]), cls


FILLERS = {"DIST": fill_dist, "TIME": fill_time, "ANGLE": fill_angle, "DIR": fill_dir, "OBJ": fill_obj}


def fill(template, intent, unknown_obj=False):
    """Fill every slot of a phrasing; return (text, label)."""
    values = {}
    text = template
    for slot in SLOTS[intent]:
        if slot == "OBJ":
            surface, val = fill_obj(unknown=unknown_obj)
        else:
            surface, val = FILLERS[slot]()
        values[slot] = val
        text = text.replace("{%s}" % slot, surface, 1)
    text = fix_articles(text)
    if unknown_obj:
        return text, "UNKNOWN"
    if intent == "TURN_PLAIN" and SPIN_RE.search(template):
        return text, "LEFT 360"       # "spin" with no amount means all the way around
    return text, LABELS[intent](values)


def fix_articles(text):
    text = re.sub(r"\b(a|an|the)\s+(a|an)\s+", r"\2 ", text, flags=re.I)
    text = re.sub(r"\b([Aa])\s+(?=(?:office|hour|egg)\b)", r"\1n ", text)
    return text


# ---------------------------------------------------------------------------
# Step 1: load and filter
# ---------------------------------------------------------------------------

def load(path):
    rows = []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            for p in rec["phrasings"]:
                rows.append({"text": p.strip(), "intent": rec["intent"],
                             "style_id": rec["style_id"], "test_only": bool(rec.get("test_only"))})
    return rows


_OBJ_WORDS = sorted({w for syns in OBJ_SYNONYMS.values() for w in syns} |
                    {"coffee mug", "cups", "bottles", "chairs"}, key=len, reverse=True)
_OBJ_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in _OBJ_WORDS) + r")\b", re.I)


def rescue_object(row):
    """'go forward until you see a bottle' -> 'go forward until you see a {OBJ}'."""
    if "OBJ" in SLOTS[row["intent"]] and "{OBJ}" not in row["text"]:
        hits = _OBJ_RE.findall(row["text"])
        if len(hits) == 1:
            row["text"] = _OBJ_RE.sub("{OBJ}", row["text"])
            row["rescued"] = True
    return row


def reject_reason(row):
    text, intent = row["text"], row["intent"]
    if not text:
        return "empty"
    if sorted(SLOT_RE.findall(text)) != sorted(SLOTS[intent]):
        return "wrong slots"
    if re.search(r"\d", text):
        return "digit"
    no_slots = SLOT_RE.sub(" ", text)
    if any(w in NUMBER_WORDS for w in words_of(no_slots)):
        return "number word"
    if len(text.split()) > 25:
        return "too long"
    if re.search(r"\ban \{OBJ\}", text, re.I):
        return "an {OBJ}"
    if intent == "CONNECTOR" and len(text.split()) > 5:
        return "keyword"
    ws = set(words_of(no_slots))
    if any(k in ws for k in KEYWORDS.get(intent, [])):
        return "keyword"
    return None


def filter_phrasings(rows, log):
    reasons = defaultdict(Counter)
    examples = defaultdict(list)
    kept = []
    rescued = 0
    for r in rows:
        r = rescue_object(r)
        rescued += bool(r.get("rescued"))
        why = reject_reason(r)
        if why:
            reasons[r["intent"]][why] += 1
            if len(examples[why]) < 20:
                examples[why].append("%s | %s" % (r["intent"], r["text"]))
        else:
            kept.append(r)

    # Dedupe within an intent, then drop phrasings that appear under two intents.
    seen, deduped = set(), []
    for r in kept:
        key = (r["intent"], normalize(r["text"]))
        if key in seen:
            reasons[r["intent"]]["duplicate"] += 1
            continue
        seen.add(key)
        deduped.append(r)
    owners = defaultdict(set)
    for r in deduped:
        owners[normalize(r["text"])].add(r["intent"])
    final = []
    for r in deduped:
        o = owners[normalize(r["text"])]
        if len(o) > 1 and o != {"WATCH_LEAD", "WATCH_TRAIL"}:
            reasons[r["intent"]]["two intents"] += 1
            if len(examples["two intents"]) < 20:
                examples["two intents"].append("%s | %s" % (r["intent"], r["text"]))
        else:
            final.append(r)

    total_in = Counter(r["intent"] for r in rows)
    total_out = Counter(r["intent"] for r in final)
    log("STEP 1  filter")
    log("  %-15s %5s %5s  %s" % ("intent", "in", "kept", "rejected by reason"))
    for intent in SLOTS:
        lost = total_in[intent] - total_out[intent]
        flag = "  <-- lost over half" if total_in[intent] and lost > total_in[intent] / 2 else ""
        log("  %-15s %5d %5d  %s%s" % (intent, total_in[intent], total_out[intent],
                                      dict(reasons[intent]), flag))
    log("  kept %d of %d (%d rescued by turning an object name into {OBJ})" % (len(final), len(rows), rescued))
    log("  sample rejects:")
    for why, ex in examples.items():
        for e in ex[:5]:
            log("    [%s] %s" % (why, e))
    return final


# ---------------------------------------------------------------------------
# Step 2: hand review
# ---------------------------------------------------------------------------

def apply_rejects(rows, log):
    if not REJECTS_PATH.exists():
        REJECTS_PATH.write_text("# Paste bad phrasings here, one per line, exactly as in review/.\n"
                                "# Lines starting with # are ignored.\n")
    bad = set()
    for line in REJECTS_PATH.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            bad.add(normalize(line.split(" | ", 1)[-1]))
    kept = [r for r in rows if normalize(r["text"]) not in bad]
    log("STEP 2  hand rejects: %d lines in rejects.txt removed %d phrasings" % (len(bad), len(rows) - len(kept)))

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    by_intent = defaultdict(list)
    for r in kept:
        by_intent[r["intent"]].append(r["text"])
    for intent, texts in by_intent.items():
        sample = rng.sample(texts, min(30, len(texts)))
        (REVIEW_DIR / ("%s.txt" % intent)).write_text("\n".join(sample) + "\n")
    log("  wrote 30 samples per intent to %s" % REVIEW_DIR.relative_to(HERE.parent))
    return kept


# ---------------------------------------------------------------------------
# Step 3: split
# ---------------------------------------------------------------------------

def split(rows, log):
    train, test = defaultdict(list), defaultdict(list)
    by_intent = defaultdict(list)
    for r in rows:
        (test[r["intent"]] if r["test_only"] else by_intent[r["intent"]]).append(r["text"])
    for intent, texts in by_intent.items():
        random.shuffle(texts)
        k = int(math.ceil(0.2 * len(texts)))
        test[intent].extend(texts[:k])
        train[intent].extend(texts[k:])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assignment = {}
    for name, d in (("train", train), ("test", test)):
        for intent, texts in d.items():
            for t in texts:
                assignment[normalize(t)] = name
    (OUT_DIR / "split.json").write_text(json.dumps(assignment, indent=0, sort_keys=True))
    log("STEP 3  split (phrasings)")
    for intent in SLOTS:
        warn = "  <-- few test phrasings" if len(test[intent]) < 5 else ""
        log("  %-15s train %4d  test %4d%s" % (intent, len(train[intent]), len(test[intent]), warn))
    return train, test


# ---------------------------------------------------------------------------
# Step 5: sequences
# ---------------------------------------------------------------------------

def strip_end(text):
    return text.rstrip().rstrip(".!?,;").rstrip()


def tidy(text):
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*,+", ",", text)
    text = re.sub(r"\bthen\s+then\b", "then", text, flags=re.I)
    text = re.sub(r"\band\s+and\b", "and", text, flags=re.I)
    return " ".join(text.split())


def join_steps(texts, connectors):
    out = strip_end(texts[0])
    for t in texts[1:]:
        if random.random() < 0.7 and connectors:
            j = strip_end(random.choice(connectors))
        else:
            j = random.choice(LIGHT_JOINERS)
        sep = "" if j.startswith(",") else " "
        out = "%s%s%s %s" % (out, sep, j, strip_end(t))
    return out


def pick_step(pool, unknown_obj=False, budget=None):
    """Pick an intent uniformly, then a phrasing, so slot-less intents aren't drowned out.
    With a character budget, retry a few times to find a phrasing that fits it."""
    best = None
    for _ in range(12):
        if unknown_obj:
            intent = random.choice(OBJ_INTENTS)
        else:
            intent = random.choice([i for i in MOVE_INTENTS if pool[i]])
        step = fill(random.choice(pool[intent]), intent, unknown_obj=unknown_obj)
        if budget is None or len(step[0]) <= budget:
            return step
        if best is None or len(step[0]) < len(best[0]):
            best = step
    return best


def pick_short(pool, intent, budget, unknown_obj=False):
    best = None
    for _ in range(12):
        c = fill(random.choice(pool[intent]), intent, unknown_obj=unknown_obj)
        if len(c[0]) <= budget:
            return c
        if best is None or len(c[0]) < len(best[0]):
            best = c
    return best


INPUT_BUDGET = 112      # chars left for the input after output tokens (about 8-16)


def make_sequence(pool, unknown_step=False):
    n = random.choices([1, 2, 3, 4], weights=[40, 35, 18, 7])[0]
    if unknown_step and n == 1:
        n = 2
    budget = INPUT_BUDGET - 3 * n                # leave room for the extra output tokens

    watch = None
    if random.random() < 0.2:                    # WATCH clause, decided first so steps leave room
        lead = random.random() < 0.5
        intent = "WATCH_LEAD" if lead else "WATCH_TRAIL"
        if pool[intent]:
            watch = (lead, pick_short(pool, intent, 45))
            budget -= len(watch[1][0]) + 1

    per_step = max(12, budget // n - 8)          # about 8 chars per joiner
    steps = [pick_step(pool, budget=per_step) for _ in range(n)]
    if unknown_step:
        steps[random.randrange(n)] = pick_step(pool, unknown_obj=True, budget=per_step)
    text = join_steps([t for t, _ in steps], pool["CONNECTOR"])
    label = "; ".join(l for _, l in steps)

    if watch:
        lead, (ctext, clabel) = watch
        if lead:
            text = strip_end(ctext) + random.choice([" ", ", "]) + text
        else:
            ctext = ctext.strip()
            text = text + ("" if ctext.startswith(",") else " ") + ctext
        label = clabel + "; " + label

    if unknown_step or "UNKNOWN" in out_tokens(label):
        label = "UNKNOWN"
    return tidy(text), label


# ---------------------------------------------------------------------------
# Step 6: refusals
# ---------------------------------------------------------------------------

def make_unknown_object(pool):
    r = random.random()
    if r < 0.5:                                   # single FWD_UNTIL / SEARCH step
        text, _ = pick_step(pool, unknown_obj=True)
        return tidy(text), "UNKNOWN"
    if r < 0.7 and pool["WATCH_LEAD"] and pool["WATCH_TRAIL"]:   # watch clause + valid steps
        steps = [pick_step(pool) for _ in range(random.choice([1, 1, 2]))]
        body = join_steps([t for t, _ in steps], pool["CONNECTOR"])
        if random.random() < 0.5:
            c, _ = fill(random.choice(pool["WATCH_LEAD"]), "WATCH_LEAD", unknown_obj=True)
            text = strip_end(c) + " " + body
        else:
            c, _ = fill(random.choice(pool["WATCH_TRAIL"]), "WATCH_TRAIL", unknown_obj=True)
            c = c.strip()
            text = body + ("" if c.startswith(",") else " ") + c
        return tidy(text), "UNKNOWN"
    return make_sequence(pool, unknown_step=True)


def make_gibberish():
    r = random.random()
    if r < 0.4:
        n = random.randint(3, 12)
        return "".join(random.choice(string.ascii_lowercase) for _ in range(n)), "UNKNOWN"
    if r < 0.6:
        row = random.choice(["qwertyuiop", "asdfghjkl", "zxcvbnm"])
        i = random.randrange(len(row) - 3)
        return row[i:i + random.randint(3, len(row) - i)], "UNKNOWN"
    if r < 0.8:
        return random.choice(["???", "...", "?", "!!!", "??!", "..", "-", "!?"]), "UNKNOWN"
    return random.choice(["hmm", "uh", "umm", "huh", "lol", "ok", "hm", "idk", "brb", "k"]), "UNKNOWN"


def make_llm_unknown(pool):
    return random.choice(pool["UNKNOWN"]).strip(), "UNKNOWN"


def make_stop(pool):
    return random.choice(pool["STOP"]).strip(), "STOP"


# ---------------------------------------------------------------------------
# Step 7: noise
# ---------------------------------------------------------------------------

def typo(word):
    i = random.randrange(len(word) - 1)
    op = random.choice(["drop", "swap", "double"])
    if op == "drop":
        return word[:i] + word[i + 1:]
    if op == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word[:i] + word[i] + word[i:]


def add_noise(text):
    text = text.lower().replace("’", "'")
    if random.random() < 0.15:
        toks = text.split(" ")
        idx = [i for i, w in enumerate(toks)
               if len(w) >= 4 and w.isalpha() and w not in NEGATIONS]
        if idx:
            i = random.choice(idx)
            toks[i] = typo(toks[i])
            text = " ".join(toks)
    if random.random() < 0.10:
        r = random.random()
        if r < 0.35:
            toks = text.split(" ")
            i = random.randrange(len(toks))
            toks[i] = toks[i] + " "
            text = " ".join(toks)
        elif r < 0.7:
            text = text.rstrip(".!?") + random.choice([".", "!", "?", "!!", ".."])
        else:
            text = text.replace(",", "")
    return text.strip()


def fits(text, label):
    return len(text) + len(out_tokens(label)) + SPECIAL_TOKENS <= MAX_TOKENS


# ---------------------------------------------------------------------------
# Step 8: balance, check, write
# ---------------------------------------------------------------------------

STEP_RE = re.compile(
    r"^(?:(?:FWD|BACK) (?:\d+|\d+(?:\.5)?s)"
    r"|FWD UNTIL (?:bottle|cup|chair|person)"
    r"|(?:LEFT|RIGHT) (?:\d+|\d+(?:\.5)?s)"
    r"|SEARCH (?:bottle|cup|chair|person))$")


def valid_label(label):
    """Grammar check, standing in for common/parser.py until it exists."""
    if label in ("STOP", "UNKNOWN"):
        return True
    parts = label.split("; ")
    if parts[0].startswith("WATCH "):
        if parts[0] not in ("WATCH bottle", "WATCH cup", "WATCH chair", "WATCH person"):
            return False
        parts = parts[1:]
    return 0 < len(parts) <= 4 and all(STEP_RE.match(p) for p in parts)


def generate(maker, n, seen, kind, kinds):
    """Make n examples with unique inputs (after noise) that fit the length cap."""
    out, tries = [], 0
    while len(out) < n and tries < 30 * n + 100:
        tries += 1
        text, label = maker()
        text = add_noise(text)
        if not text or text in seen or not fits(text, label):
            continue
        seen.add(text)
        out.append({"input": text, "output": label})
        kinds[kind] += 1
    return out


def build_set(pool, size, log, name):
    # Shares from the guide: STOP ~3%, UNKNOWN ~11% (LLM 5%, unknown objects 4%, gibberish 2%).
    seen, kinds = set(), Counter()
    plan = [
        ("stop", lambda: make_stop(pool), int(0.03 * size)),
        ("unknown_llm", lambda: make_llm_unknown(pool), int(0.05 * size)),
        ("unknown_object", lambda: make_unknown_object(pool), int(0.04 * size)),
        ("gibberish", make_gibberish, int(0.02 * size)),
    ]
    data = []
    for kind, maker, n in plan:
        got = generate(maker, n, seen, kind, kinds)
        if len(got) < n:
            log("  [%s] %s: only %d unique of %d wanted" % (name, kind, len(got), n))
            if kind.startswith("unknown") or kind == "gibberish":   # top up refusals from objects
                got += generate(lambda: make_unknown_object(pool), n - len(got), seen, "unknown_object", kinds)
        data += got
    data += generate(lambda: make_sequence(pool), size - len(data), seen, "commands", kinds)
    random.shuffle(data)
    return data, kinds


def check_and_write(train, test, log):
    errors = []

    for name, data in (("train", train), ("test", test)):
        bad = [d for d in data if not valid_label(d["output"])]
        if bad:
            errors.append("%s: %d labels fail the grammar check, e.g. %s" % (name, len(bad), bad[:3]))
        long_ = [d for d in data if not fits(d["input"], d["output"])]
        if long_:
            errors.append("%s: %d examples over the length cap" % (name, len(long_)))

    def drop_conflicts(data, name):
        outs = defaultdict(set)
        for d in data:
            outs[d["input"]].add(d["output"])
        conflict = {i for i, o in outs.items() if len(o) > 1}
        if conflict:
            log("  %s: dropped %d inputs with two different outputs" % (name, len(conflict)))
        uniq, seen = [], set()
        for d in data:
            key = (d["input"], d["output"])
            if d["input"] in conflict or key in seen:
                continue
            seen.add(key)
            uniq.append(d)
        return uniq

    train = drop_conflicts(train, "train")
    test = drop_conflicts(test, "test")
    train_inputs = {d["input"] for d in train}
    overlap = [d for d in test if d["input"] in train_inputs]
    if overlap:
        log("  removed %d test examples whose input also appears in train" % len(overlap))
        test = [d for d in test if d["input"] not in train_inputs]
    assert not ({d["input"] for d in test} & train_inputs)

    vocab = Counter()
    for d in train + test:
        vocab.update(out_tokens(d["output"]))

    if errors:
        for e in errors:
            log("ERROR " + e)
        sys.exit("Checks failed; nothing written.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in (("train", train), ("test", test)):
        with (OUT_DIR / ("%s.jsonl" % name)).open("w") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    (OUT_DIR / "output_vocab.txt").write_text("\n".join(sorted(vocab)) + "\n")
    return train, test, vocab


def describe(data):
    c = Counter()
    steps = Counter()
    for d in data:
        o = d["output"]
        if o in ("STOP", "UNKNOWN"):
            c[o] += 1
            continue
        parts = o.split("; ")
        if parts[0].startswith("WATCH"):
            c["with WATCH"] += 1
            parts = parts[1:]
        steps[len(parts)] += 1
        for p in parts:
            c[p.split()[0] + (" UNTIL" if " UNTIL " in p else "")] += 1
    return c, steps


# ---------------------------------------------------------------------------
# Step 9: human test set
# ---------------------------------------------------------------------------

def check_human_test(train, log):
    if not HUMAN_TEST_PATH.exists():
        log("STEP 9  no %s yet (write it by hand; see the guide)" % HUMAN_TEST_PATH.name)
        return
    rows = [json.loads(l) for l in HUMAN_TEST_PATH.read_text().splitlines() if l.strip()]
    bad = [r for r in rows if not valid_label(r["output"])]
    train_inputs = {d["input"] for d in train}
    leaked = [r for r in rows if r["input"].lower() in train_inputs]
    log("STEP 9  human test set: %d commands, %d bad labels, %d also in train"
        % (len(rows), len(bad), len(leaked)))
    for r in bad[:10]:
        log("    bad label: %s" % r)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-size", type=int, default=40000)
    ap.add_argument("--test-size", type=int, default=4000)
    args = ap.parse_args()

    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    rows = load(RAW_PATH)
    rows = filter_phrasings(rows, log)
    rows = apply_rejects(rows, log)
    train_pool, test_pool = split(rows, log)

    log("STEP 4-7  fill, sequences, refusals, noise")
    train, train_kinds = build_set(train_pool, args.train_size, log, "train")
    test, test_kinds = build_set(test_pool, args.test_size, log, "test")

    log("STEP 8  check and write")
    train, test, vocab = check_and_write(train, test, log)
    for name, data, kinds in (("train", train, train_kinds), ("test", test, test_kinds)):
        c, steps = describe(data)
        n = len(data)
        log("  %s: %d examples" % (name, n))
        log("    sources: %s" % dict(kinds))
        log("    STOP %.1f%%  UNKNOWN %.1f%%  with WATCH %.1f%%" % (
            100.0 * c["STOP"] / n, 100.0 * c["UNKNOWN"] / n, 100.0 * c["with WATCH"] / n))
        log("    steps per command: %s" % dict(sorted(steps.items())))
        log("    step actions: %s" % {k: v for k, v in c.items() if k not in ("STOP", "UNKNOWN", "with WATCH")})
    log("  output vocabulary: %d tokens (data/out/output_vocab.txt)" % len(vocab))
    log("  20 random training examples:")
    for d in random.sample(train, min(20, len(train))):
        log("    %-70s -> %s" % (d["input"][:70], d["output"]))

    check_human_test(train, log)
    (OUT_DIR / "stats.txt").write_text("\n".join(lines) + "\n")
    log("Done. Now read 50 random lines of data/out/train.jsonl yourself.")


if __name__ == "__main__":
    main()
