"""Shared loader: the exact list of (set_name, member_label, text) scored by score.py."""
import json, os, re

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def clean_fresh(t):
    # Pile Wikipedia text has no "== Heading ==" lines; strip them so the
    # fresh set differs from the Pile set in date, not in markup.
    lines = [l for l in t.split("\n") if not re.match(r"^\s*=+.*=+\s*$", l)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def load_sets():
    tr = json.load(open(os.path.join(D, "pile_train_sample.json")))["domains"]
    te = json.load(open(os.path.join(D, "pile_test_sample.json")))["domains"]
    fr = json.load(open(os.path.join(D, "wikipedia_fresh_2026.json")))["articles"]
    sets = []
    for dom, key in [("Wikipedia (en)", "wiki"), ("ArXiv", "arxiv"), ("Pile-CC", "cc")]:
        sets += [(f"pile_train_{key}", 1, t) for t in tr[dom]]
        sets += [(f"pile_test_{key}", 0, t) for t in te[dom]]
    sets += [("fresh_wiki_2026", 0, clean_fresh(a["text"])) for a in fr]
    for L in (64, 128):
        for r in json.load(open(os.path.join(D, f"wikimia_{L}.json")))["rows"]:
            sets.append((f"wikimia_{L}", r["label"], r["text"]))
    return sets
