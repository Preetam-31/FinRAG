
import sys
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import json
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# Import your existing ingestion functions
from ingestion import extract_text_from_pdf, create_chunks


def main():
    print("Starting retrieval test...")

    # Load document chunks
    pdf_path = ROOT / "data" / "documents" / "company_report.pdf"

    print("Loading PDF:", pdf_path)

    pages = extract_text_from_pdf(str(pdf_path))

    print("Total pages:", len(pages))

    chunks = create_chunks(pages)

    print("Total chunks:", len(chunks))

    print("Total chunks:", len(chunks))

    # Load retrieval models
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Loading reranker...")
    reranker = CrossEncoder(
        "BAAI/bge-reranker-base"
    )

    # Connect to local Qdrant
    client = QdrantClient(path=str(ROOT / "qdrant_data"))

    collection_name = "finrag_documents"

    # Load evaluation questions
    questions_path = ROOT / "evaluation" / "questions.json"

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Prepare BM25
    tokenized_chunks = [
        chunk["text"].lower().split()
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    for item in questions:
        question = item["question"]

        print("\n" + "=" * 70)
        print("QUESTION:", question)

        # 1. BM25 retrieval
        tokenized_question = question.lower().split()

        bm25_scores = bm25.get_scores(tokenized_question)

        bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:10]

        # 2. Vector retrieval
        query_vector = embedding_model.encode(
            question
        ).tolist()

        vector_results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=10,
            with_payload=True
        ).points

        vector_indices = []

        for result in vector_results:
            payload = result.payload or {}

            if "chunk_index" in payload:
                index = int(payload["chunk_index"])
            elif "index" in payload:
                index = int(payload["index"])
            else:
                try:
                    index = int(result.id)
                except (ValueError, TypeError):
                    continue

            if 0 <= index < len(chunks):
                vector_indices.append(index)

        # 3. Reciprocal Rank Fusion
        rrf_scores = {}

        for rank, index in enumerate(bm25_indices):
            rrf_scores[index] = (
                rrf_scores.get(index, 0) + 1 / (60 + rank + 1)
            )

        for rank, index in enumerate(vector_indices):
            rrf_scores[index] = (
                rrf_scores.get(index, 0) + 1 / (60 + rank + 1)
            )

        candidate_indices = sorted(
            rrf_scores,
            key=rrf_scores.get,
            reverse=True
        )[:10]

        # 4. Cross-encoder reranking
        pairs = [
            (question, chunks[index]["text"])
            for index in candidate_indices
        ]

        if pairs:
            scores = reranker.predict(pairs)

            reranked = sorted(
                zip(candidate_indices, scores),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            print("\nTOP 5 RETRIEVED CHUNKS:")

            for rank, (index, score) in enumerate(reranked, 1):
                chunk = chunks[index]

                print("\nRank:", rank)
                print("Chunk index:", index)
                print("Reranker score:", round(float(score), 4))
                print("Page:", chunk.get("page_number", "Unknown"))
                print("Text:", chunk["text"][:1000])

        else:
            print("No chunks retrieved.")

    client.close()

    print("\nRetrieval test completed.")


if __name__ == "__main__":
    main()
