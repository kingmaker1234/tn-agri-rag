"""Pipeline 1 · stage 1 — LOAD.

Turns any supported file into a list of `Section`s (text + location metadata such
as page / sheet / slide). Every loader is defensive: a broken file produces an
error message, never a crash of the whole ingestion job.

Supported: pdf, docx, doc*, odt*, rtf*, pptx, ppt*, odp*, xlsx, xlsm, xls, ods*,
csv, tsv, html/htm/asp/aspx/php/xhtml, txt, md, json, xml, png/jpg/jpeg/tif/tiff/bmp/webp (OCR),
zip (expanded by the pipeline).  (* = converted with LibreOffice when installed)
"""
from __future__ import annotations

import io
import json
import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ..config import settings
from ..text_utils import clean_text, fix_tamil_visual_order, legacy_font_ratio

log = logging.getLogger(__name__)


@dataclass
class Section:
    text: str
    meta: dict = field(default_factory=dict)  # page, sheet, slide, heading ...


@dataclass
class LoadedDocument:
    sections: list[Section]
    title: str = ""
    n_pages: int = 0
    notes: list[str] = field(default_factory=list)  # warnings shown in the UI
    extraction: str = "text"  # text | ocr | mixed | converted


class LoaderError(Exception):
    pass


NOTE_TAMIL_ORDER = "Tamil letters re-ordered (PDF stored them in visual order)"


HTML_EXT = {".html", ".htm", ".xhtml", ".asp", ".aspx", ".php", ".jsp"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
OFFICE_CONVERT = {".doc": "docx", ".odt": "docx", ".rtf": "docx", ".ppt": "pptx", ".odp": "pptx", ".ods": "xlsx"}
SUPPORTED_EXT = (
    {".pdf", ".docx", ".pptx", ".xlsx", ".xlsm", ".xls", ".csv", ".tsv", ".txt", ".md", ".json", ".xml", ".zip"}
    | HTML_EXT | IMAGE_EXT | set(OFFICE_CONVERT)
)


# --------------------------------------------------------------------------- OCR
_ocr_state: dict = {}


def ocr_available() -> bool:
    if settings.ocr_enabled == "false":
        return False
    if "ok" not in _ocr_state:
        try:
            import pytesseract  # noqa: F401

            _ocr_state["ok"] = shutil.which("tesseract") is not None
        except Exception:
            _ocr_state["ok"] = False
    return _ocr_state["ok"]


def _ocr_image_bytes(png_bytes: bytes) -> str:
    import pytesseract
    from PIL import Image

    img = Image.open(io.BytesIO(png_bytes))
    return pytesseract.image_to_string(img, lang=settings.ocr_languages)


# --------------------------------------------------------------------------- helpers
def _read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16"):
        try:
            text = raw.decode(enc)
            if enc == "utf-16" and not raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
                raise UnicodeDecodeError(enc, b"", 0, 1, "no bom")
            return text
        except UnicodeDecodeError:
            continue
    try:
        import chardet

        enc = chardet.detect(raw[:200000]).get("encoding") or "latin-1"
        return raw.decode(enc, errors="replace")
    except Exception:
        return raw.decode("latin-1", errors="replace")


def _post(text: str, notes: list[str]) -> str:
    text, fixed = fix_tamil_visual_order(text)
    if fixed and NOTE_TAMIL_ORDER not in notes:
        notes.append(NOTE_TAMIL_ORDER)
    return clean_text(text)


def _rows_to_text(header: list[str], rows: list[list], max_rows: int) -> list[str]:
    """Render table rows as self-contained 'Header: value' lines so that every
    chunk keeps its column context (crucial for retrieval over spreadsheets)."""
    lines = []
    header = [str(h).strip() if h is not None and str(h).strip() and not str(h).startswith("Unnamed") else f"col{i+1}"
              for i, h in enumerate(header)]
    for r in rows[:max_rows]:
        cells = []
        for h, v in zip(header, r):
            if v is None:
                continue
            s = str(v).strip()
            if not s or s.lower() == "nan":
                continue
            cells.append(f"{h}: {s}")
        if cells:
            lines.append(" | ".join(cells))
    return lines


# --------------------------------------------------------------------------- loaders
def load_pdf(path: Path) -> LoadedDocument:
    import pymupdf

    notes: list[str] = []
    sections: list[Section] = []
    ocr_pages = 0
    try:
        doc = pymupdf.open(str(path))
    except Exception as e:  # encrypted / corrupt
        raise LoaderError(f"cannot open PDF: {e}")
    if doc.needs_pass:
        raise LoaderError("PDF is password protected")
    title = (doc.metadata or {}).get("title") or ""
    legacy_pages = 0
    for i, page in enumerate(doc):
        text = page.get_text("text", sort=True) or ""
        legacy = legacy_font_ratio(text) > 0.15
        legacy_pages += legacy
        if (len(text.strip()) < 40 or legacy) and ocr_available():
            try:
                pix = page.get_pixmap(dpi=200)
                o = _ocr_image_bytes(pix.tobytes("png"))
                if len(o.strip()) > len(text.strip()) * 0.5:
                    text = o
                    ocr_pages += 1
            except Exception as e:  # pragma: no cover
                log.warning("OCR failed on %s p%d: %s", path.name, i + 1, e)
        text = _post(text, notes)
        if text:
            sections.append(Section(text, {"page": i + 1}))
    n_pages = len(doc)
    doc.close()
    if legacy_pages and not ocr_pages:
        notes.append(f"{legacy_pages} page(s) look like a legacy Tamil font; install Tesseract with Tamil data for OCR")
    if not sections:
        notes.append("No text layer found (scanned PDF?); install Tesseract to OCR it")
    extraction = "ocr" if ocr_pages == n_pages and n_pages else ("mixed" if ocr_pages else "text")
    return LoadedDocument(sections, title, n_pages, notes, extraction)


def load_docx(path: Path) -> LoadedDocument:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    notes: list[str] = []
    d = docx.Document(str(path))
    title = d.core_properties.title or ""
    sections: list[Section] = []
    buf: list[str] = []
    heading = ""

    def flush():
        if buf:
            t = _post("\n".join(buf), notes)
            if t:
                sections.append(Section(t, {"heading": heading} if heading else {}))
            buf.clear()

    body = d.element.body
    for child in body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, d)
            txt = p.text.strip()
            if not txt:
                continue
            style = (p.style.name or "").lower() if p.style is not None else ""
            if style.startswith("heading") or style == "title":
                flush()
                heading = txt
                if not title and style == "title":
                    title = txt
            buf.append(txt)
        elif tag == "tbl":
            t = Table(child, d)
            rows = [[c.text.strip() for c in r.cells] for r in t.rows]
            if rows:
                buf.extend(_rows_to_text(rows[0], rows[1:], settings.max_table_rows) or [" | ".join(rows[0])])
    flush()
    return LoadedDocument(sections, title, 0, notes)


def load_pptx(path: Path) -> LoadedDocument:
    from pptx import Presentation

    notes: list[str] = []
    prs = Presentation(str(path))
    sections = []
    title = prs.core_properties.title or ""
    for i, slide in enumerate(prs.slides, 1):
        parts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                parts.append(shape.text_frame.text)
            if getattr(shape, "has_table", False) and shape.has_table:
                rows = [[c.text for c in r.cells] for r in shape.table.rows]
                if rows:
                    parts.extend(_rows_to_text(rows[0], rows[1:], 500))
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            n = slide.notes_slide.notes_text_frame.text.strip()
            if n:
                parts.append("Notes: " + n)
        text = _post("\n".join(parts), notes)
        if text:
            sections.append(Section(text, {"slide": i}))
    return LoadedDocument(sections, title, len(prs.slides), notes)


def load_spreadsheet(path: Path) -> LoadedDocument:
    import pandas as pd

    notes: list[str] = []
    ext = path.suffix.lower()
    engine = "xlrd" if ext == ".xls" else None
    try:
        sheets = pd.read_excel(path, sheet_name=None, header=None, engine=engine, dtype=str)
    except Exception as e:
        raise LoaderError(f"cannot read spreadsheet: {e}")
    sections = []
    for name, df in sheets.items():
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if df.empty:
            continue
        # header = first row that has >= half the columns filled
        hdr_idx = 0
        for idx in range(min(10, len(df))):
            if df.iloc[idx].notna().sum() >= max(1, df.shape[1] // 2):
                hdr_idx = idx
                break
        header = list(df.iloc[hdr_idx].fillna("").astype(str))
        rows = df.iloc[hdr_idx + 1:].values.tolist()
        if len(rows) > settings.max_table_rows:
            notes.append(f"Sheet “{name}” truncated to {settings.max_table_rows} rows")
        lines = [f"Sheet: {name}"] + _rows_to_text(header, rows, settings.max_table_rows)
        text = _post("\n".join(lines), notes)
        if text:
            sections.append(Section(text, {"sheet": str(name)}))
    return LoadedDocument(sections, "", len(sheets), notes)


def load_csv(path: Path) -> LoadedDocument:
    import pandas as pd

    notes: list[str] = []
    text = _read_text_file(path)
    sep = "\t" if path.suffix.lower() == ".tsv" else None
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, engine="python", dtype=str, on_bad_lines="skip")
    except Exception as e:
        raise LoaderError(f"cannot parse CSV: {e}")
    lines = _rows_to_text(list(df.columns), df.values.tolist(), settings.max_table_rows)
    if len(df) > settings.max_table_rows:
        notes.append(f"Truncated to {settings.max_table_rows} rows")
    return LoadedDocument([Section(_post("\n".join(lines), notes), {})], "", 0, notes)


_HTML_DROP = ["script", "style", "noscript", "svg", "nav", "footer", "header", "form", "iframe", "button", "aside"]


def load_html(path: Path) -> LoadedDocument:
    from bs4 import BeautifulSoup

    notes: list[str] = []
    html = _read_text_file(path)
    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True) and (not title or len(title) < 8):
        title = h1.get_text(" ", strip=True)
    for tag in soup(_HTML_DROP):
        tag.decompose()
    for el in soup.select('[class*="menu"], [id*="menu"], [class*="breadcrumb"], [class*="cookie"], [class*="social"]'):
        el.decompose()
    # tables -> 'Header: value' rows
    for table in soup.find_all("table"):
        rows = [[c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])] for tr in table.find_all("tr")]
        rows = [r for r in rows if any(r)]
        if len(rows) >= 2 and max(len(r) for r in rows) > 1:
            lines = _rows_to_text(rows[0], rows[1:], 2000)
            table.replace_with(soup.new_string("\n" + "\n".join(lines) + "\n"))
    main = soup.find("main") or soup.find("article") or soup.find(id=re.compile("content|main", re.I)) or soup.body or soup
    for br in main.find_all(["br"]):
        br.replace_with("\n")
    for blk in main.find_all(["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section"]):
        blk.insert_after("\n")
    text = _post(main.get_text(" "), notes)
    # Angular/React shells (e.g. myscheme.gov.in) ship almost no server-side text
    if len(text) < 200:
        notes.append("Very little text in this page (it may be rendered by JavaScript)")
    return LoadedDocument([Section(text, {})] if text else [], title, 0, notes)


def load_text(path: Path) -> LoadedDocument:
    notes: list[str] = []
    text = _read_text_file(path)
    title = ""
    if path.suffix.lower() == ".md":
        m = re.search(r"^#\s+(.+)$", text, re.M)
        title = m.group(1).strip() if m else ""
        # split markdown on headings so sections keep their heading
        parts = re.split(r"(?m)^(?=#{1,3}\s)", text)
        secs = []
        for p in parts:
            hm = re.match(r"#{1,3}\s+(.+)", p)
            p = re.sub(r"(?m)^\s{0,3}(#{1,6}|>)\s?", "", p)  # drop markdown heading / quote markers
            t = _post(p, notes)
            if t:
                secs.append(Section(t, {"heading": hm.group(1).strip()} if hm else {}))
        return LoadedDocument(secs, title, 0, notes)
    return LoadedDocument([Section(_post(text, notes), {})], title, 0, notes)


def _flatten_json(obj, prefix="") -> list[str]:
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_flatten_json(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        if obj and all(isinstance(x, dict) for x in obj):
            for i, x in enumerate(obj):
                flat = _flatten_json(x)
                out.append(f"{prefix}[{i}] " + " | ".join(flat))
        else:
            for i, x in enumerate(obj):
                out.extend(_flatten_json(x, f"{prefix}[{i}]"))
    elif obj is not None and str(obj).strip():
        out.append(f"{prefix}: {obj}")
    return out


def load_json(path: Path) -> LoadedDocument:
    notes: list[str] = []
    raw = _read_text_file(path)
    try:
        data = json.loads(raw)
    except Exception:
        return load_text(path)
    lines = _flatten_json(data)
    return LoadedDocument([Section(_post("\n".join(lines), notes), {})], "", 0, notes)


def load_xml(path: Path) -> LoadedDocument:
    from bs4 import BeautifulSoup

    notes: list[str] = []
    soup = BeautifulSoup(_read_text_file(path), "lxml-xml")
    return LoadedDocument([Section(_post(soup.get_text("\n"), notes), {})], "", 0, notes)


def load_image(path: Path) -> LoadedDocument:
    if not ocr_available():
        raise LoaderError("image needs OCR: install tesseract-ocr (with Tamil 'tam' data) and `pip install pytesseract pillow`")
    notes: list[str] = []
    text = _ocr_image_bytes(path.read_bytes())
    return LoadedDocument([Section(_post(text, notes), {"page": 1})], "", 1, notes, "ocr")


def _soffice_convert(path: Path, target: str) -> Path | None:
    exe = shutil.which("soffice") or shutil.which("libreoffice")
    if not exe:
        return None
    out_dir = Path(tempfile.mkdtemp(prefix="conv_"))
    try:
        subprocess.run([exe, "--headless", "--convert-to", target, "--outdir", str(out_dir), str(path)],
                       check=True, timeout=180, capture_output=True)
        produced = list(out_dir.glob(f"*.{target}"))
        return produced[0] if produced else None
    except Exception as e:  # pragma: no cover
        log.warning("soffice conversion failed for %s: %s", path, e)
        return None


def _legacy_doc_fallback(path: Path) -> LoadedDocument:
    """Last resort for .doc when LibreOffice is missing or fails: pull readable runs."""
    has_office = bool(shutil.which("soffice") or shutil.which("libreoffice"))
    notes = ["LibreOffice could not convert this file; basic text recovery used" if has_office
             else "Basic text recovery used; install LibreOffice for full .doc support"]
    if shutil.which("antiword"):
        try:
            out = subprocess.run(["antiword", "-w", "0", str(path)], capture_output=True, timeout=60, check=True)
            text = out.stdout.decode("utf-8", errors="replace")
            return LoadedDocument([Section(_post(text, []), {})], "", 0, ["Text extracted with antiword"], "converted")
        except Exception:
            pass
    raw = path.read_bytes()
    utf16 = raw.decode("utf-16le", errors="ignore")
    runs16 = [r for r in re.findall(r"[A-Za-z0-9஀-௿][A-Za-z0-9஀-௿ \.,;:()/%&'\-–]{6,}", utf16)
              if sum(c.isalpha() for c in r) >= 4]
    runs8 = [r.decode("latin-1") for r in re.findall(rb"[\x20-\x7e]{6,}", raw)]
    runs = runs16 if sum(map(len, runs16)) >= sum(map(len, runs8)) else runs8
    text = "\n".join(runs)
    return LoadedDocument([Section(_post(text, notes), {})], "", 0, notes, "converted")


def load_file(path: Path) -> LoadedDocument:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    if ext == ".docx":
        return load_docx(path)
    if ext == ".pptx":
        return load_pptx(path)
    if ext in (".xlsx", ".xlsm", ".xls"):
        return load_spreadsheet(path)
    if ext in (".csv", ".tsv"):
        return load_csv(path)
    if ext in HTML_EXT:
        return load_html(path)
    if ext in (".txt", ".md"):
        return load_text(path)
    if ext == ".json":
        return load_json(path)
    if ext == ".xml":
        return load_xml(path)
    if ext in IMAGE_EXT:
        return load_image(path)
    if ext in OFFICE_CONVERT:
        target = OFFICE_CONVERT[ext]
        converted = _soffice_convert(path, target)
        if converted:
            doc = load_file(converted)
            doc.extraction = "converted"
            doc.notes.append(f"Converted from .{ext[1:]} with LibreOffice")
            shutil.rmtree(converted.parent, ignore_errors=True)
            return doc
        if ext == ".doc":
            return _legacy_doc_fallback(path)
        raise LoaderError(f"{ext} needs LibreOffice (soffice) for conversion")
    raise LoaderError(f"unsupported file type: {ext or '(none)'}")


def sniff_extension(path: Path) -> str:
    """Guess a real extension for files saved without one (common with gov sites)."""
    head = path.read_bytes()[:2048]
    if head.startswith(b"%PDF"):
        return ".pdf"
    if head.startswith(b"PK"):
        with open(path, "rb") as f:
            data = f.read(200000)
        if b"word/" in data:
            return ".docx"
        if b"ppt/" in data:
            return ".pptx"
        if b"xl/" in data:
            return ".xlsx"
        return ".zip"
    if head.startswith(b"\xd0\xcf\x11\xe0"):
        return ".doc"  # OLE2 (doc/xls/ppt) – doc is the common case
    low = head.lower()
    if b"<html" in low or b"<!doctype html" in low:
        return ".html"
    if head.lstrip()[:1] in (b"{", b"["):
        return ".json"
    if head.startswith(b"\x89PNG") or head[:3] == b"\xff\xd8\xff":
        return ".png" if head.startswith(b"\x89PNG") else ".jpg"
    return ".txt"
