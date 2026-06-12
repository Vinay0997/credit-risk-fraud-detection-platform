"""Citation-backed regulatory and policy assistant."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedChunk:
    source: str
    chunk_id: int
    text: str
    score: float


class PolicyAssistant:
    """Local retrieval assistant for demo and audit-safe development."""

    def __init__(self, docs_path: str | Path = "data/regulations") -> None:
        self.docs_path = Path(docs_path)
        self.chunks = self._load_chunks()
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([chunk.text for chunk in self.chunks])

    def _load_chunks(self) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        for doc_path in sorted(self.docs_path.glob("*.md")):
            text = doc_path.read_text(encoding="utf-8")
            paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
            for index, paragraph in enumerate(paragraphs):
                chunks.append(
                    RetrievedChunk(
                        source=doc_path.name,
                        chunk_id=index,
                        text=paragraph,
                        score=0.0,
                    )
                )
        if not chunks:
            raise ValueError(f"No markdown policy documents found in {self.docs_path}")
        return chunks

    def retrieve(self, question: str, top_k: int = 3) -> list[RetrievedChunk]:
        question_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(question_vector, self.matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        return [
            RetrievedChunk(
                source=self.chunks[index].source,
                chunk_id=self.chunks[index].chunk_id,
                text=self.chunks[index].text,
                score=float(scores[index]),
            )
            for index in top_indices
            if scores[index] > 0
        ]

    def answer(self, question: str, top_k: int = 3) -> dict[str, object]:
        matches = self.retrieve(question, top_k=top_k)
        if not matches:
            return {
                "answer": "I could not find enough supporting policy evidence to answer that question.",
                "citations": [],
            }

        evidence_lines = []
        citations = []
        for match in matches:
            evidence_lines.append(match.text)
            citations.append(
                {
                    "source": match.source,
                    "chunk_id": match.chunk_id,
                    "relevance": round(match.score, 4),
                }
            )
        return {
            "answer": " ".join(evidence_lines),
            "citations": citations,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--docs", type=Path, default=Path("data/regulations"))
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assistant = PolicyAssistant(args.docs)
    response = assistant.answer(args.question, top_k=args.top_k)
    print(response["answer"])
    if response["citations"]:
        print("\nCitations:")
        for citation in response["citations"]:
            print(
                f"- {citation['source']}#chunk-{citation['chunk_id']} "
                f"(relevance={citation['relevance']})"
            )


if __name__ == "__main__":
    main()
