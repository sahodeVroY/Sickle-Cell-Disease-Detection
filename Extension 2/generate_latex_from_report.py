from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path("/mnt/c/Users/sahod/Desktop/project")
MARKDOWN_PATH = ROOT / "Proposed_System_Report.md"
OUTPUT_PATH = ROOT / "Proposed_System_Chapter.tex"


@dataclass
class Heading:
    level: int
    text: str


@dataclass
class Paragraph:
    text: str


@dataclass
class Table:
    caption: str
    rows: list[list[str]]


@dataclass
class Figure:
    caption: str
    path: str


Block = Heading | Paragraph | Table | Figure


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "item"


def strip_leading_numbering(text: str) -> str:
    return re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", text).strip()


def strip_figure_table_prefix(text: str) -> str:
    return re.sub(r"^(Figure|Table)\s+\d+\.?\s*:?\s*", "", text, flags=re.IGNORECASE).strip()


def escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    text = "".join(replacements.get(ch, ch) for ch in text)
    text = text.replace("×", r"$\times$")
    text = text.replace("→", r"$\rightarrow$")
    text = text.replace("≤", r"$\leq$")
    text = text.replace("≥", r"$\geq$")
    text = text.replace(r"\textasciitilde{}", "approximately ")
    text = re.sub(r"(\d+)x(\d+)", r"\1$\\times$\2", text)
    text = text.replace("->", r"$\rightarrow$")
    text = text.replace(".keras", r"\texttt{.keras}")
    text = text.replace("—", "--")
    text = text.replace("–", "--")
    return text


def parse_markdown(lines: list[str]) -> list[Block]:
    blocks: list[Block] = []
    pending_table_caption = ""
    idx = 0

    while idx < len(lines):
        line = lines[idx].strip()

        if not line:
            idx += 1
            continue

        if line.startswith("**") and line.endswith("**"):
            caption = line.strip("*").strip()
            if caption.lower().startswith("table "):
                pending_table_caption = caption
            else:
                blocks.append(Paragraph(caption))
            idx += 1
            continue

        if line.startswith("![") and "](" in line and line.endswith(")"):
            caption = strip_figure_table_prefix(line[2:].split("](", 1)[0].strip())
            path = line.split("](", 1)[1][:-1].strip()
            relative_path = Path(path).relative_to(ROOT).as_posix()
            blocks.append(Figure(caption=caption, path=relative_path))
            idx += 1
            if idx < len(lines):
                next_line = lines[idx].strip()
                if next_line.startswith("*") and next_line.endswith("*"):
                    idx += 1
            continue

        if line.startswith("|") and line.endswith("|"):
            table_lines = [line]
            idx += 1
            while idx < len(lines):
                nxt = lines[idx].strip()
                if nxt.startswith("|") and nxt.endswith("|"):
                    table_lines.append(nxt)
                    idx += 1
                else:
                    break
            rows = parse_table(table_lines)
            blocks.append(Table(caption=pending_table_caption, rows=rows))
            pending_table_caption = ""
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading_match:
            blocks.append(Heading(level=len(heading_match.group(1)), text=heading_match.group(2).strip()))
            idx += 1
            continue

        blocks.append(Paragraph(line))
        idx += 1

    return blocks


def parse_table(lines: Iterable[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw in lines:
        cells = [cell.strip() for cell in raw.strip("|").split("|")]
        if all(set(cell.replace(" ", "")) <= {"-"} for cell in cells):
            continue
        rows.append(cells)
    return rows


def latex_heading(block: Heading) -> str:
    command = {
        1: "chapter",
        2: "section",
        3: "subsection",
        4: "subsubsection",
        5: "paragraph",
        6: "subparagraph",
    }[block.level]
    text = strip_leading_numbering(block.text)
    if text.rstrip(":") == "Workflow Description":
        return f"\\paragraph{{{escape_latex(text.rstrip(':'))}}}"
    return f"\\{command}{{{escape_latex(text)}}}"


def latex_paragraph(block: Paragraph) -> str:
    raw_text = block.text.strip()
    text = escape_latex(raw_text)

    if raw_text.endswith(":") and len(raw_text.split()) <= 6:
        return f"\\paragraph{{{escape_latex(raw_text[:-1])}}}"

    label_match = re.match(r"^([^:]{1,60}):\s+(.*)$", raw_text)
    if label_match:
        label, rest = label_match.groups()
        return f"\\noindent\\textbf{{{escape_latex(label)}:}} {escape_latex(rest)}"

    return text


def latex_table(block: Table) -> str:
    col_count = max(len(row) for row in block.rows)
    colspec = "|" + "|".join(["p{0.22\\textwidth}"] * col_count) + "|"
    fallback_caption = "Dataset Structure" if block.rows and block.rows[0][:2] == ["Folder", "Class"] else "Table"
    caption_text = strip_figure_table_prefix(block.caption or fallback_caption)
    caption = escape_latex(caption_text)
    label = slugify(caption_text)

    rows = []
    for row_idx, row in enumerate(block.rows):
        padded = row + [""] * (col_count - len(row))
        cells = [escape_latex(cell) for cell in padded]
        if row_idx == 0:
            cells = [rf"\textbf{{{cell}}}" for cell in cells]
        rows.append(" & ".join(cells) + r" \\ \hline")

    return "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{tab:{label}}}",
            rf"\begin{{tabular}}{{{colspec}}}",
            r"\hline",
            *rows,
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def latex_figure(block: Figure) -> str:
    caption = escape_latex(strip_figure_table_prefix(block.caption))
    label = slugify(strip_figure_table_prefix(block.caption))
    return "\n".join(
        [
            r"\begin{figure}[htbp]",
            r"\centering",
            rf"\includegraphics[width=\textwidth]{{{block.path}}}",
            rf"\caption{{{caption}}}",
            rf"\label{{fig:{label}}}",
            r"\end{figure}",
        ]
    )


def render_latex(blocks: list[Block]) -> str:
    preamble = [
        "% Required packages in your Overleaf preamble:",
        "% \\usepackage{graphicx}",
        "% \\usepackage{float}",
        "% \\usepackage{array}",
        "",
    ]

    rendered: list[str] = []
    for block in blocks:
        if isinstance(block, Heading):
            rendered.append(latex_heading(block))
        elif isinstance(block, Paragraph):
            rendered.append(latex_paragraph(block))
        elif isinstance(block, Table):
            rendered.append(latex_table(block))
        elif isinstance(block, Figure):
            rendered.append(latex_figure(block))
        rendered.append("")

    return "\n".join(preamble + rendered).rstrip() + "\n"


def main() -> None:
    lines = MARKDOWN_PATH.read_text(encoding="utf-8").splitlines()
    blocks = parse_markdown(lines)
    latex = render_latex(blocks)
    OUTPUT_PATH.write_text(latex, encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
