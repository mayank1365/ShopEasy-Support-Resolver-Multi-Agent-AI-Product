"""Retrieval-Augmented Generation over the help-doc knowledge base.

Why RAG here: the Drafter must answer using ShopEasy's *actual* policies
(30-day window, $200 refund approval, etc.), not whatever the model guesses.
We retrieve the most relevant help-doc chunks and feed them to the Drafter as
grounding — this is what keeps replies accurate and auditable.

Implementation: lightweight TF-IDF + cosine similarity (scikit-learn). This is a
real vector-space retriever with no external API or embedding key required, which
keeps the project self-contained and easy to run for a demo.
"""

from __future__ import annotations

import functools
import pathlib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_DIR = pathlib.Path(__file__).parent / "knowledge_base"


def _load_chunks() -> list[tuple[str, str]]:
    """Load every markdown doc and split it into section chunks.

    Returns a list of (source_filename, chunk_text). We split on the '## '
    markdown headers so each chunk is a self-contained policy section.
    """
    chunks: list[tuple[str, str]] = []
    for path in sorted(KB_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        # Split on level-2 headers; keep the header with its body.
        sections = text.split("\n## ")
        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
            # Re-attach the '## ' we removed (except the very first block).
            chunk = section if i == 0 else "## " + section
            chunks.append((path.name, chunk))
    return chunks


@functools.lru_cache(maxsize=1)
def _build_index():
    """Build the TF-IDF index once and cache it for the process lifetime."""
    chunks = _load_chunks()
    corpus = [c for _, c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    return chunks, vectorizer, matrix


def retrieve(query: str, k: int = 3) -> list[tuple[str, str]]:
    """Return the top-k (source, chunk) pairs most relevant to the query.

    Chunks with zero lexical overlap are filtered out, so an off-topic query
    can legitimately return fewer than k (or zero) results.
    """
    chunks, vectorizer, matrix = _build_index()
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix)[0]
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results: list[tuple[str, str]] = []
    for idx, score in ranked[:k]:
        if score <= 0.0:
            continue
        results.append(chunks[idx])
    return results
