"""Assemble paper/paper.md from individual section files.

Re-runnable; idempotent. Header levels are normalized so the title is the
single H1, section files (which start with `# Section N: ...`) become H2,
their subsections (`## N.X`) become H3, etc.

Run: PYTHONNOUSERSITE=1 python -m scripts.build_paper
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TITLE = "Token-F1 Inverts Answer Correctness: An Audit of Long-Horizon Dialogue Memory Evaluation"
SUBTITLE = "COLM 2026 submission draft"
ORDER = [
    "abstract.md",
    "section_1_introduction.md",
    "section_2_related_work.md",
    "section_3_method.md",
    "section_4_setup.md",
    "section_5_results.md",
    "section_6_discussion.md",
    "section_7_limitations.md",
    "section_8_conclusion.md",
]


def demote(md: str) -> str:
    out = []
    for line in md.splitlines():
        if re.match(r"^_Draft v\d.*$", line.strip()):
            continue
        if line.startswith("#### "):
            out.append("##### " + line[5:])
        elif line.startswith("### "):
            out.append("#### " + line[4:])
        elif line.startswith("## "):
            out.append("### " + line[3:])
        elif line.startswith("# "):
            out.append("## " + line[2:])
        else:
            out.append(line)
    return "\n".join(out).strip() + "\n"


def main():
    parts = [f"# {TITLE}\n", f"_{SUBTITLE}_\n", "---\n"]
    for fn in ORDER:
        body = (ROOT / "paper" / fn).read_text()
        parts.append(demote(body))
        parts.append("\n---\n")
    out_path = ROOT / "paper" / "paper.md"
    out_path.write_text("\n".join(parts))
    n_lines = sum(1 for _ in out_path.read_text().splitlines())
    print(f"wrote {out_path} ({n_lines} lines)")


if __name__ == "__main__":
    main()
