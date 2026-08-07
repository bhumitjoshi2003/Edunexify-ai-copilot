"""
services/chunker.py — splits extracted document text into overlapping chunks
for embedding. Hand-written, paragraph/sentence-aware — no LangChain.

Sentences are packed greedily into ~chunk_size-character chunks; when a chunk
fills up, the last ~overlap characters' worth of sentences carry over into the
next chunk so a fact split across a chunk boundary still has context on both sides.
"""
import re

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sentences: list[str] = []
    for para in paragraphs:
        sentences.extend(s.strip() for s in _SENTENCE_SPLIT_RE.split(para) if s.strip())

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        if current and current_len + len(sentence) + 1 > chunk_size:
            chunks.append(" ".join(current))

            # Carry the tail of this chunk forward for continuity across the boundary.
            overlap_sentences: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s) + 1
            current = overlap_sentences
            current_len = overlap_len

        current.append(sentence)
        current_len += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))

    # A single sentence longer than chunk_size (rare — e.g. a huge table dumped
    # as one line) would otherwise become its own oversized chunk. Hard-split
    # as a fallback so nothing embeds a multi-thousand-character blob.
    final: list[str] = []
    for c in chunks:
        if len(c) <= chunk_size * 1.5:
            final.append(c)
        else:
            final.extend(c[i:i + chunk_size] for i in range(0, len(c), chunk_size))
    return final
