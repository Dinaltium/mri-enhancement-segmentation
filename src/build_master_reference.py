"""
build_master_reference.py -- assemble ONE file containing everything.

The project's knowledge is spread across several documents, which is right for
writing but wrong for a viva: under questioning you want a single file you can
search. This concatenates them, adds a table of contents, and appends a numbers
appendix generated FROM results/*.json rather than retyped -- so the appendix
cannot drift out of date the way a hand-maintained list would.

Run:  python src/build_master_reference.py
Out:  docs/MASTER_REFERENCE.md
"""

import json
import os
import re
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = "docs/MASTER_REFERENCE.md"

# order matters: quick answers first, depth after, definitions last
PARTS = [
    ("PART I — VIVA: EVERY QUESTION AND ITS ANSWER", "docs/VIVA_PREP.md"),
    ("PART II — WHAT WE BUILT, IN FULL DETAIL", "docs/KNOWLEDGE_BASE.md"),
    ("PART III — WHY WE USED A PRETRAINED MODEL", "docs/PRETRAINED_MODEL_JUSTIFICATION.md"),
    ("PART IV — GLOSSARY: EVERY TERM", "docs/GLOSSARY.md"),
    ("PART V — WHAT EACH STAGE ASKED FOR", "docs/STAGES.md"),
]


def demote(md: str, by: int = 1) -> str:
    """Push every heading down a level so the merged file has one clean tree."""
    out = []
    for line in md.splitlines():
        if line.startswith("#") and not line.startswith("#" * 7):
            line = "#" * by + line
        out.append(line)
    return "\n".join(out)


def flatten(obj, prefix=""):
    """Walk a results JSON and yield (path, scalar) for every leaf number."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        if obj and all(isinstance(x, (int, float)) for x in obj):
            yield prefix, f"[{len(obj)} values]"
        else:
            for i, v in enumerate(obj[:4]):
                yield from flatten(v, f"{prefix}[{i}]")
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        yield prefix, obj


def numbers_appendix() -> str:
    """Every measured value we hold, straight from the JSON files."""
    import glob
    lines = ["", "# PART VI — EVERY MEASURED NUMBER WE HOLD", "",
             "Generated directly from `results/*.json`. If a judge asks for a figure",
             "not in Parts I–V, it is almost certainly here. Nothing in this section",
             "was retyped by hand, so it cannot disagree with the code.", ""]
    for p in sorted(glob.glob("results/*.json")):
        try:
            with open(p) as f:
                data = json.load(f)
        except Exception:
            continue
        rows = list(flatten(data))
        if not rows:
            continue
        lines += [f"## `{os.path.basename(p)}`", "", "| Field | Value |", "|---|---|"]
        for k, v in rows[:70]:
            if isinstance(v, float):
                v = round(v, 4)
            sv = str(v)
            if len(sv) > 80:
                sv = sv[:77] + "..."
            lines.append(f"| `{k}` | {sv} |")
        if len(rows) > 70:
            lines.append(f"| … | *{len(rows) - 70} more fields in the file* |")
        lines.append("")
    return "\n".join(lines)


def file_inventory() -> str:
    """What every source file does — asked as 'explain your codebase'."""
    import glob
    lines = ["", "# PART VII — EVERY SOURCE FILE, AND WHAT IT DOES", "",
             "One line each, taken from the file's own docstring.", "",
             "| File | Purpose |", "|---|---|"]
    for p in sorted(glob.glob("src/*.py") + glob.glob("demos/*.py")):
        doc = ""
        try:
            with open(p, encoding="utf-8") as f:
                src = f.read(4000)
            m = re.search(r'"""(.*?)"""', src, re.S)
            if m:
                body = [l.strip() for l in m.group(1).strip().splitlines() if l.strip()]
                # skip a bare "modulename.py" first line
                for l in body:
                    if not l.lower().startswith(os.path.basename(p).lower().split(".")[0]):
                        doc = l
                        break
                doc = doc or (body[0] if body else "")
        except Exception:
            pass
        doc = doc.replace("|", "\\|")[:150]
        lines.append(f"| `{p}` | {doc} |")
    return "\n".join(lines)


def main():
    n_src = len([f for f in os.listdir("src") if f.endswith(".py")])
    head = [
        "# MASTER REFERENCE — everything, in one file",
        "",
        "*Single-file consolidation of every document in this project, plus every*",
        "*measured number, plus what every source file does. Built by*",
        "*`src/build_master_reference.py` — re-run it after any change.*",
        "",
        "**How to use this under questioning.** Search it (Ctrl+F) rather than reading",
        "it. Part I is the fast path: it holds the twelve numbers to memorise and the",
        "direct answers. Part VI holds every figure we have ever measured, so if you",
        "are asked something specific it is almost certainly there.",
        "",
        "**The one rule:** if you cannot find a number, say *\"I don't have that",
        "memorised — it's in `results/`, I can pull it up now\"*. That reads as rigour.",
        "Inventing a number is the only losing move.",
        "",
        "---",
        "",
        "## Contents",
        "",
        "| Part | Contains |",
        "|---|---|",
        "| **I** | Viva prep — the 12 key numbers, loss functions, inputs, every likely question |",
        "| **II** | Full detail — architectures, every training run, every method, what we ditched |",
        "| **III** | The pretrained-model justification, as given to the organisers |",
        "| **IV** | Glossary — every term, in plain words |",
        "| **V** | What each of the four stages asked for |",
        "| **VI** | Every measured number, generated from `results/*.json` |",
        f"| **VII** | What each of the {n_src} source files does |",
        "",
        "---",
        "",
    ]

    body = []
    for title, path in PARTS:
        if not os.path.exists(path):
            print(f"  ! missing {path}, skipped")
            continue
        with open(path, encoding="utf-8") as f:
            md = f.read()
        body += ["", f"# {title}", "",
                 f"*Source: `{path}`*", "", demote(md), "", "---", ""]
        print(f"  + {path}")

    text = "\n".join(head + body) + "\n" + numbers_appendix() + "\n" + file_inventory() + "\n"
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    words = len(text.split())
    print(f"\n  wrote {OUT}  ({len(text.splitlines())} lines, ~{words:,} words)")


if __name__ == "__main__":
    main()
