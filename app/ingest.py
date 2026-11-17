"""Pipeline de ingestão: extração -> limpeza -> chunking -> embeddings -> ChromaDB.

Uso: python ingest.py [--if-empty]
"""
import glob
import os
import re
import sys
from collections import Counter

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pypdf import PdfReader

import rag

DOC_DIR = os.getenv("DOC_DIR", os.path.join(os.path.dirname(__file__), "..", "doc"))

# chunk_size=800 / overlap=120 (15%): as seções "##" dos .md têm ~400-1500 caracteres, então 800
# mantém a maioria das tabelas de período/pré-requisitos inteiras, fica abaixo do limite de 512
# tokens do encoder, e o overlap evita cortar uma linha "código + pré-requisito" entre dois chunks.
CHUNK_SIZE, CHUNK_OVERLAP = 800, 120
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", "\n- ", "\n", ". ", " "]
)
md_splitter = MarkdownHeaderTextSplitter([("#", "title"), ("##", "section")], strip_headers=False)


def clean(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # hífen de quebra de linha
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def doc_type(fname: str) -> str:
    f = fname.lower()
    return "matriz" if "matriz" in f else "guia" if "guia" in f else "grade" if "grade" in f else "outro"


def topic_for(dtype: str, section: str, text: str) -> str:
    """Topic key do chunk: usado pelas tools MCP para filtrar o grupo de embeddings."""
    if "complementar" in section.lower() or "atividades complementares" in text.lower():
        return "horas_complementares"
    if dtype == "matriz":  # matriz curricular = disciplinas, períodos, pré-requisitos
        return "pre_requisitos"
    return "geral"


def load_chunks(path: str) -> list[tuple[str, dict]]:
    fname = os.path.basename(path)
    dtype = doc_type(fname)
    pieces: list[tuple[str, str]] = []  # (section, text)

    if path.endswith(".md"):
        raw = open(path, encoding="utf-8").read()
        raw = re.sub(r"\A---\n.*?\n---\n", "", raw, flags=re.S)  # remove front-matter YAML
        for d in md_splitter.split_text(raw):
            section = d.metadata.get("section", d.metadata.get("title", ""))
            pieces += [(section, c) for c in splitter.split_text(clean(d.page_content))]
    else:
        text = clean("\n".join(p.extract_text() or "" for p in PdfReader(path).pages))
        if len(text) < 200:
            print(f"  [aviso] {fname}: sem texto extraível (PDF escaneado), ignorado — use a versão .md")
            return []
        pieces = [("", c) for c in splitter.split_text(text)]

    return [
        (text, {"source": fname, "doc_type": dtype, "section": section, "chunk_index": i,
                "topic": topic_for(dtype, section, text)})
        for i, (section, text) in enumerate(pieces)
    ]


def main():
    col = rag.collection()
    if "--if-empty" in sys.argv and col.count() > 0:
        print(f"Coleção já possui {col.count()} chunks; ingestão ignorada.")
        return
    rag.client().delete_collection(rag.COLLECTION)
    col = rag.collection()

    chunks = []
    for path in sorted(glob.glob(os.path.join(DOC_DIR, "*.md")) + glob.glob(os.path.join(DOC_DIR, "*.pdf"))):
        c = load_chunks(path)
        print(f"  {os.path.basename(path)}: {len(c)} chunks")
        chunks += c

    texts = [t for t, _ in chunks]
    metas = [m for _, m in chunks]
    ids = [f"{m['source']}::{m['chunk_index']}" for m in metas]
    for i in range(0, len(texts), 64):
        col.add(ids=ids[i:i + 64], documents=texts[i:i + 64], metadatas=metas[i:i + 64],
                embeddings=rag.embed(texts[i:i + 64]))
    print(f"Total: {len(texts)} chunks | por topic: {dict(Counter(m['topic'] for m in metas))}")


if __name__ == "__main__":
    main()
