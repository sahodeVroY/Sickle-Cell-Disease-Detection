from __future__ import annotations

import html
import itertools
import re
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path("/mnt/c/Users/sahod/Desktop/project")
MARKDOWN_PATH = ROOT / "Proposed_System_Report.md"
OUTPUT_PATH = ROOT / "Proposed_System_Report.docx"


XML_HEADER = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
EMU_PER_PIXEL = 9525
MAX_IMAGE_WIDTH_EMU = 6 * 914400


@dataclass
class Paragraph:
    text: str
    style: str = "Normal"
    italic: bool = False
    bold: bool = False
    center: bool = False


@dataclass
class Table:
    rows: list[list[str]]


@dataclass
class ImageBlock:
    path: Path
    caption: str | None = None


def escape_xml(text: str) -> str:
    return html.escape(text, quote=False)


def read_png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        header = f.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Unsupported image format for {path}")
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def markdown_to_blocks(text: str) -> list[Paragraph | Table | ImageBlock]:
    lines = text.splitlines()
    blocks: list[Paragraph | Table | ImageBlock] = []
    idx = 0

    while idx < len(lines):
        line = lines[idx].rstrip()
        stripped = line.strip()

        if not stripped:
            idx += 1
            continue

        if stripped.startswith("![") and "](" in stripped and stripped.endswith(")"):
            alt = stripped[2:].split("](", 1)[0]
            path_text = stripped.split("](", 1)[1][:-1]
            caption = None
            next_idx = idx + 1
            while next_idx < len(lines) and not lines[next_idx].strip():
                next_idx += 1
            if next_idx < len(lines):
                maybe_caption = lines[next_idx].strip()
                if maybe_caption.startswith("*") and maybe_caption.endswith("*") and len(maybe_caption) > 2:
                    caption = maybe_caption[1:-1].strip()
                    idx = next_idx
            blocks.append(ImageBlock(path=Path(path_text), caption=caption or alt))
            idx += 1
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines = [stripped]
            idx += 1
            while idx < len(lines):
                next_line = lines[idx].strip()
                if next_line.startswith("|") and next_line.endswith("|"):
                    table_lines.append(next_line)
                    idx += 1
                else:
                    break
            rows = parse_markdown_table(table_lines)
            if rows:
                blocks.append(Table(rows=rows))
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            style = {
                1: "Title",
                2: "Heading1",
                3: "Heading2",
                4: "Heading3",
                5: "Heading4",
                6: "Heading4",
            }[level]
            blocks.append(Paragraph(text=heading_text, style=style))
            idx += 1
            continue

        if stripped.startswith("**") and stripped.endswith("**") and len(stripped) > 4:
            blocks.append(Paragraph(text=stripped[2:-2].strip(), style="Heading3"))
            idx += 1
            continue

        if stripped.startswith("*") and stripped.endswith("*") and len(stripped) > 2:
            blocks.append(Paragraph(text=stripped[1:-1].strip(), style="Caption", italic=True, center=True))
            idx += 1
            continue

        blocks.append(Paragraph(text=stripped))
        idx += 1

    return blocks


def parse_markdown_table(lines: Iterable[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw in lines:
        cells = [cell.strip() for cell in raw.strip().strip("|").split("|")]
        if all(set(cell) <= {"-"} for cell in cells):
            continue
        rows.append(cells)
    return rows


def paragraph_xml(paragraph: Paragraph) -> str:
    p_pr = []
    if paragraph.style != "Normal":
        p_pr.append(f'<w:pStyle w:val="{paragraph.style}"/>')
    if paragraph.center:
        p_pr.append('<w:jc w:val="center"/>')
    p_pr_xml = f"<w:pPr>{''.join(p_pr)}</w:pPr>" if p_pr else ""

    text = paragraph.text.replace("\t", "    ")
    run_props = []
    if paragraph.bold:
        run_props.append("<w:b/>")
    if paragraph.italic:
        run_props.append("<w:i/>")
    r_pr_xml = f"<w:rPr>{''.join(run_props)}</w:rPr>" if run_props else ""
    run_xml = f"<w:r>{r_pr_xml}<w:t xml:space=\"preserve\">{escape_xml(text)}</w:t></w:r>"
    return f"<w:p>{p_pr_xml}{run_xml}</w:p>"


def table_xml(table: Table) -> str:
    col_count = max(len(row) for row in table.rows) if table.rows else 1
    tbl_grid = "".join('<w:gridCol w:w="2400"/>' for _ in range(col_count))
    rows_xml = []

    for row_index, row in enumerate(table.rows):
        row = row + [""] * (col_count - len(row))
        cell_xml = []
        for cell in row:
            p = Paragraph(text=cell, bold=(row_index == 0))
            cell_xml.append(
                "<w:tc>"
                "<w:tcPr><w:tcW w:w=\"2400\" w:type=\"dxa\"/></w:tcPr>"
                f"{paragraph_xml(p)}"
                "</w:tc>"
            )
        rows_xml.append("<w:tr>" + "".join(cell_xml) + "</w:tr>")

    return (
        "<w:tbl>"
        "<w:tblPr>"
        "<w:tblW w:w=\"0\" w:type=\"auto\"/>"
        "<w:tblBorders>"
        "<w:top w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:left w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:bottom w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:right w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:insideH w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:insideV w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"auto\"/>"
        "</w:tblBorders>"
        "</w:tblPr>"
        f"<w:tblGrid>{tbl_grid}</w:tblGrid>"
        f"{''.join(rows_xml)}"
        "</w:tbl>"
    )


def image_paragraph_xml(rel_id: str, image_name: str, width_px: int, height_px: int, docpr_id: int) -> str:
    width_emu = width_px * EMU_PER_PIXEL
    height_emu = height_px * EMU_PER_PIXEL
    if width_emu > MAX_IMAGE_WIDTH_EMU:
        scale = MAX_IMAGE_WIDTH_EMU / width_emu
        width_emu = int(width_emu * scale)
        height_emu = int(height_emu * scale)

    return f"""
<w:p>
  <w:pPr><w:jc w:val="center"/></w:pPr>
  <w:r>
    <w:drawing>
      <wp:inline distT="0" distB="0" distL="0" distR="0"
        xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">
        <wp:extent cx="{width_emu}" cy="{height_emu}"/>
        <wp:docPr id="{docpr_id}" name="{escape_xml(image_name)}"/>
        <wp:cNvGraphicFramePr>
          <a:graphicFrameLocks noChangeAspect="1"
            xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"/>
        </wp:cNvGraphicFramePr>
        <a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
          <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
              <pic:nvPicPr>
                <pic:cNvPr id="{docpr_id}" name="{escape_xml(image_name)}"/>
                <pic:cNvPicPr/>
              </pic:nvPicPr>
              <pic:blipFill>
                <a:blip r:embed="{rel_id}"/>
                <a:stretch><a:fillRect/></a:stretch>
              </pic:blipFill>
              <pic:spPr>
                <a:xfrm>
                  <a:off x="0" y="0"/>
                  <a:ext cx="{width_emu}" cy="{height_emu}"/>
                </a:xfrm>
                <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
              </pic:spPr>
            </pic:pic>
          </a:graphicData>
        </a:graphic>
      </wp:inline>
    </w:drawing>
  </w:r>
</w:p>
""".strip()


def build_document(blocks: list[Paragraph | Table | ImageBlock]) -> tuple[str, str, dict[str, bytes]]:
    body_xml: list[str] = []
    relationships: list[str] = []
    media: dict[str, bytes] = {}
    image_counter = itertools.count(1)
    docpr_counter = itertools.count(1)

    for block in blocks:
        if isinstance(block, Paragraph):
            body_xml.append(paragraph_xml(block))
        elif isinstance(block, Table):
            body_xml.append(table_xml(block))
        elif isinstance(block, ImageBlock):
            image_index = next(image_counter)
            rel_id = f"rId{image_index}"
            ext = block.path.suffix.lower()
            media_name = f"image{image_index}{ext}"
            relationships.append(
                f'<Relationship Id="{rel_id}" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
                f'Target="media/{media_name}"/>'
            )
            media[f"word/media/{media_name}"] = block.path.read_bytes()
            width_px, height_px = read_png_size(block.path)
            body_xml.append(
                image_paragraph_xml(rel_id, media_name, width_px, height_px, next(docpr_counter))
            )
            if block.caption:
                body_xml.append(
                    paragraph_xml(
                        Paragraph(text=block.caption, style="Caption", italic=True, center=True)
                    )
                )

    body_xml.append(
        "<w:sectPr>"
        "<w:pgSz w:w=\"12240\" w:h=\"15840\"/>"
        "<w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\" "
        "w:header=\"720\" w:footer=\"720\" w:gutter=\"0\"/>"
        "</w:sectPr>"
    )

    document_xml = (
        XML_HEADER
        + '<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
        'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
        'xmlns:v="urn:schemas-microsoft-com:vml" '
        'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:w10="urn:schemas-microsoft-com:office:word" '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
        'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
        'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
        'xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" '
        'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
        'mc:Ignorable="w14 wp14">'
        f"<w:body>{''.join(body_xml)}</w:body></w:document>"
    )

    rels_xml = (
        XML_HEADER
        + '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rIdStyles" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
        + "".join(relationships)
        + "</Relationships>"
    )

    return document_xml, rels_xml, media


def styles_xml() -> str:
    return (
        XML_HEADER
        + '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:style w:type="paragraph" w:default="1" w:styleId="Normal">'
        '<w:name w:val="Normal"/>'
        '<w:qFormat/>'
        '<w:rPr><w:sz w:val="22"/></w:rPr>'
        '</w:style>'
        '<w:style w:type="paragraph" w:styleId="Title">'
        '<w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:qFormat/>'
        '<w:pPr><w:jc w:val="center"/><w:spacing w:after="240"/></w:pPr>'
        '<w:rPr><w:b/><w:sz w:val="36"/></w:rPr>'
        '</w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading1">'
        '<w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:qFormat/>'
        '<w:pPr><w:spacing w:before="240" w:after="120"/></w:pPr>'
        '<w:rPr><w:b/><w:sz w:val="30"/></w:rPr>'
        '</w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading2">'
        '<w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:qFormat/>'
        '<w:pPr><w:spacing w:before="180" w:after="100"/></w:pPr>'
        '<w:rPr><w:b/><w:sz w:val="26"/></w:rPr>'
        '</w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading3">'
        '<w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:qFormat/>'
        '<w:pPr><w:spacing w:before="120" w:after="80"/></w:pPr>'
        '<w:rPr><w:b/><w:sz w:val="24"/></w:rPr>'
        '</w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading4">'
        '<w:name w:val="heading 4"/><w:basedOn w:val="Normal"/><w:qFormat/>'
        '<w:pPr><w:spacing w:before="100" w:after="60"/></w:pPr>'
        '<w:rPr><w:b/><w:sz w:val="22"/></w:rPr>'
        '</w:style>'
        '<w:style w:type="paragraph" w:styleId="Caption">'
        '<w:name w:val="Caption"/><w:basedOn w:val="Normal"/><w:qFormat/>'
        '<w:pPr><w:jc w:val="center"/><w:spacing w:before="60" w:after="140"/></w:pPr>'
        '<w:rPr><w:i/><w:sz w:val="20"/></w:rPr>'
        '</w:style>'
        "</w:styles>"
    )


def content_types_xml(media: dict[str, bytes]) -> str:
    defaults = [
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
    ]
    image_exts = sorted({Path(name).suffix.lower().lstrip(".") for name in media})
    for ext in image_exts:
        if ext == "png":
            defaults.append('<Default Extension="png" ContentType="image/png"/>')
        elif ext in {"jpg", "jpeg"}:
            defaults.append('<Default Extension="jpg" ContentType="image/jpeg"/>')

    overrides = [
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>',
        '<Override PartName="/word/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>',
    ]
    return (
        XML_HEADER
        + '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        + "".join(defaults)
        + "".join(overrides)
        + "</Types>"
    )


def package_rels_xml() -> str:
    return (
        XML_HEADER
        + '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/>'
        "</Relationships>"
    )


def write_docx(output_path: Path, document_xml: str, rels_xml: str, media: dict[str, bytes]) -> None:
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types_xml(media))
        docx.writestr("_rels/.rels", package_rels_xml())
        docx.writestr("word/document.xml", document_xml)
        docx.writestr("word/_rels/document.xml.rels", rels_xml)
        docx.writestr("word/styles.xml", styles_xml())
        for name, data in media.items():
            docx.writestr(name, data)


def main() -> None:
    markdown = MARKDOWN_PATH.read_text(encoding="utf-8")
    blocks = markdown_to_blocks(markdown)
    document_xml, rels_xml, media = build_document(blocks)
    write_docx(OUTPUT_PATH, document_xml, rels_xml, media)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
