"""Bring-your-own-document RAG.

1. Extract text from an uploaded PDF / Word / text file (page by page for PDFs,
   so answers can cite page numbers).
2. Chunk it with the chosen strategy.
3. Embed into its own in-memory VectorStore (kept separate from the demo data).

Uploads live in memory only and disappear when the server restarts.
"""
import io
import re
import uuid
from pathlib import Path

from backend import config
from backend.rag.chunkers import Chunk, chunk_fixed_size, chunk_paragraph, chunk_sentence
from backend.rag.vector_store import VectorStore, build_store

ALLOWED = {".pdf", ".docx", ".txt", ".md"}
MAX_BYTES = 20 * 1024 * 1024
PARAGRAPH_TARGET = 800      # merge short paragraphs up to about this many characters

_UPLOADS: dict[str, dict] = {}          # doc_id -> {"name", "pages": [(page_no, text)]}
_STORES: dict[tuple, VectorStore] = {}  # (doc_id, strategy, size, overlap, embedder) -> store


# --------------------------------------------------------------------------
# Text extraction
# --------------------------------------------------------------------------
def extract_pdf(data: bytes) -> list[tuple[int, str]]:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    return [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, start=1)]


def extract_docx(data: bytes) -> list[tuple[int, str]]:
    """Word has no pages in the file format, so we return one 'page'.
    Headings become '## ' lines so the paragraph chunker can follow the structure."""
    import docx
    doc = docx.Document(io.BytesIO(data))
    lines = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        style = (p.style.name or "").lower() if p.style is not None else ""
        if style == "title":
            lines.append(f"# {text}")
        elif style.startswith("heading"):
            lines.append(f"\n## {text}")
        else:
            lines.append(text + "\n")
    for table in doc.tables:   # tables -> "cell | cell | cell" rows
        for row in table.rows:
            lines.append(" | ".join(c.text.strip() for c in row.cells) + "\n")
    return [(1, "\n".join(lines))]


def extract(filename: str, data: bytes) -> list[tuple[int, str]]:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED:
        raise ValueError(f"Unsupported file type '{ext}'. Use PDF, Word (.docx), .txt or .md "
                         "(old .doc files: open in Word and 'Save as .docx').")
    if len(data) > MAX_BYTES:
        raise ValueError("File is larger than 20 MB.")
    if ext == ".pdf":
        pages = extract_pdf(data)
    elif ext == ".docx":
        pages = extract_docx(data)
    else:
        pages = [(1, data.decode("utf-8", errors="replace"))]
    if not any(t.strip() for _, t in pages):
        raise ValueError("No text found. If this is a scanned PDF it needs OCR first.")
    return pages


# --------------------------------------------------------------------------
# Chunking for arbitrary documents
# --------------------------------------------------------------------------
def chunk_blank_lines(name: str, text: str, target: int = PARAGRAPH_TARGET) -> list[Chunk]:
    """For documents without '## ' headings (most PDFs): split on blank lines and
    merge neighbouring short paragraphs until a chunk is about `target` characters.
    PDF text often has no blank lines at all, so then we split on line breaks instead."""
    parts = re.split(r"\n\s*\n", text)
    if len(parts) <= 1:
        parts = text.split("\n")
    paras = [" ".join(p.split()) for p in parts if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if buf and len(buf) + len(p) > target:
            chunks.append(buf)
            buf = p
        else:
            buf = f"{buf} {p}".strip()
    if buf:
        chunks.append(buf)
    # A PDF page with no blank lines at all becomes one long chunk: fall back to fixed size.
    out = []
    for i, c in enumerate(chunks):
        if len(c) > target * 2:
            out += chunk_fixed_size(f"{name}-{i}", c, size=target, overlap=100)
        else:
            out.append(Chunk(f"{name}#para-{i}", c, name, "paragraph", {}))
    return out


def chunk_document(name: str, pages: list[tuple[int, str]], strategy: str,
                   size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page_no, text in pages:
        if not text.strip():
            continue
        label = f"{name} p.{page_no}" if len(pages) > 1 else name
        if strategy == "fixed_size":
            part = chunk_fixed_size(label, text, size=size, overlap=overlap)
        elif strategy == "sentence":
            part = chunk_sentence(label, text, n=config.SENTENCES_PER_CHUNK)
        elif "\n## " in "\n" + text:
            part = chunk_paragraph(label, text)       # Word doc with headings
        else:
            part = chunk_blank_lines(label, text)
        for c in part:
            c.source = label
            c.metadata["page"] = page_no
        chunks += part
    # make ids unique and short
    for i, c in enumerate(chunks):
        c.id = f"{c.source}#{i}"
    return chunks


# --------------------------------------------------------------------------
# Upload registry
# --------------------------------------------------------------------------
def add_upload(filename: str, data: bytes) -> dict:
    pages = extract(filename, data)
    doc_id = uuid.uuid4().hex[:8]
    _UPLOADS[doc_id] = {"name": Path(filename).name, "pages": pages}
    return describe(doc_id)


def describe(doc_id: str) -> dict:
    u = _UPLOADS[doc_id]
    full = "\n".join(t for _, t in u["pages"])
    return {"doc_id": doc_id, "name": u["name"], "pages": len(u["pages"]),
            "chars": len(full), "preview": full[:600]}


def list_uploads() -> list[dict]:
    return [describe(d) for d in _UPLOADS]


def remove_upload(doc_id: str) -> None:
    _UPLOADS.pop(doc_id)
    for key in [k for k in _STORES if k[0] == doc_id]:
        del _STORES[key]


def get_upload_store(doc_id: str, strategy: str = "paragraph", size: int = config.FIXED_CHUNK_SIZE,
                     overlap: int = config.FIXED_CHUNK_OVERLAP, embedder: str = "minilm") -> VectorStore:
    if doc_id not in _UPLOADS:
        raise ValueError("Unknown document. Upload it again (uploads are cleared on server restart).")
    key = (doc_id, strategy, size, overlap, embedder)
    if key not in _STORES:
        u = _UPLOADS[doc_id]
        chunks = chunk_document(u["name"], u["pages"], strategy, size, overlap)
        _STORES[key] = build_store(chunks, embedder)
    return _STORES[key]
