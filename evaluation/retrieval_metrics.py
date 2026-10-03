import os
import json
import sys
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

QUESTIONS_PATH = ROOT_DIR / "evaluation" / "questions.json"
PDF_PATH = ROOT_DIR / "data" / "documents" / "company_report.pdf"
OUTPUT_PATH = ROOT_DIR / "evaluation" / "retrieval_metrics.csv"


# ============================================================
# 2. LOAD ENVIRONMENT
# ============================================================

load_dotenv(ROOT_DIR / ".env")


# ============================================================
# 3. LOAD DOCUMENT
# ============================================================

sys.path.insert(0, str(ROOT_DIR / "src"))

from ingestion import extract_text_from_pdf, create_chunks


print("=" * 70)
print("FINRAG RETRIEVAL EVALUATION")
print("=" * 70)


# ============================================================
# 4. LOAD QUESTIONS
# ============================================================

if not QUESTIONS_PATH.exists():
    raise SystemExit(
        f"Questions file not found: {QUESTIONS_PATH}"
    )


with QUESTIONS_PATH.open(
    "r",
    encoding="utf-8"
) as file:

    questions = json.load(file)


print("\nNumber of evaluation questions:", len(questions))


# ============================================================
# 5. LOAD PDF AND CREATE CHUNKS
# ============================================================

if not PDF_PATH.exists():
    raise SystemExit(
        f"PDF file not found: {PDF_PATH}"
    )


print("\nLoading PDF...")

pages = extract_text_from_pdf(
    str(PDF_PATH)
)

chunks = create_chunks(pages)


print("Number of document chunks:", len(chunks))


if not chunks:
    raise SystemExit(
        "No document chunks were created."
    )


# ============================================================
# 6. CREATE BM25 INDEX
# ============================================================

print("\nCreating BM25 index...")


documents = [
    chunk["text"]
    for chunk in chunks
]


tokenized_documents = [
    document.lower().split()
    for document in documents
]


bm25 = BM25Okapi(
    tokenized_documents
)


# ============================================================
# 7. LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")


embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 8. LOAD CROSS-ENCODER
# ============================================================

print("\nLoading Cross-Encoder...")


reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)


# ============================================================
# 9. CONNECT TO QDRANT
# ============================================================

print("\nConnecting to Qdrant...")


qdrant_client = QdrantClient(
    path=str(ROOT_DIR / "qdrant_data")
)


collection_name = "finrag_documents"


# ============================================================
# 10. FIND EXPECTED VALUE FROM REFERENCE
# ============================================================

def get_expected_value(reference):
    """
    Extract the main financial value from the reference answer.

    Example:
    "Apple's net income in 2025 was $112,010 million."

    returns:
    "112010"
    """

    match = re.search(
        r"\$([\d,]+(?:\.\d+)?)",
        reference
    )

    if match:
        return match.group(1).replace(",", "")

    return None


# ============================================================
# 11. FIND RELEVANT CHUNKS
# ============================================================

def find_relevant_chunks(reference, chunks):
    """
    Find chunks containing the expected answer value.

    This gives us a simple ground-truth signal for
    this financial QA benchmark.
    """

    expected_value = get_expected_value(reference)

    if expected_value is None:
        return []

    relevant_indices = []

    for index, chunk in enumerate(chunks):

        text = chunk["text"]

        # Remove commas so that:
        # 416,161
        # and
        # 416161
        # can both match.

        normalized_text = text.replace(",", "")

        if expected_value in normalized_text:

            relevant_indices.append(index)

    return relevant_indices


# ============================================================
# 12. EVALUATION
# ============================================================

results = []


for question_number, question_data in enumerate(
    questions,
    start=1
):

    question = question_data["question"]
    reference = question_data["reference"]


    print("\n" + "=" * 70)
    print(
        f"QUESTION {question_number} OF {len(questions)}"
    )
    print("=" * 70)

    print("Question:", question)


    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    expected_value = get_expected_value(
        reference
    )

    relevant_chunks = find_relevant_chunks(
        reference,
        chunks
    )


    print(
        "Expected value:",
        expected_value
    )

    print(
        "Relevant chunks found:",
        len(relevant_chunks)
    )


    # --------------------------------------------------------
    # Start retrieval timer
    # --------------------------------------------------------

    retrieval_start = time.perf_counter()


    # ========================================================
    # BM25 RETRIEVAL
    # ========================================================

    query_tokens = question.lower().split()


    bm25_scores = bm25.get_scores(
        query_tokens
    )


    bm25_top_indices = (
        bm25_scores
        .argsort()[-20:][::-1]
    )


    # ========================================================
    # VECTOR RETRIEVAL
    # ========================================================

    query_embedding = embedding_model.encode(
        question
    ).tolist()


    vector_results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=10,
        with_payload=True
    ).points


    # ========================================================
    # BM25 RANKS
    # ========================================================

    bm25_rank = {

        int(index): rank + 1

        for rank, index in enumerate(
            bm25_top_indices
        )
    }


    # ========================================================
    # VECTOR RANKS
    # ========================================================

    vector_rank = {}


    for rank, result in enumerate(
        vector_results
    ):

        try:

            point_index = int(
                result.id
            )

        except (TypeError, ValueError):

            payload = result.payload or {}

            point_index = payload.get(
                "chunk_index",
                payload.get("index")
            )


            if point_index is None:
                continue


            point_index = int(
                point_index
            )


        if 0 <= point_index < len(chunks):

            vector_rank[
                point_index
            ] = rank + 1


    # ========================================================
    # RECIPROCAL RANK FUSION
    # ========================================================

    hybrid_scores = {}


    all_indices = (
        set(bm25_rank)
        |
        set(vector_rank)
    )


    for index in all_indices:

        score = 0.0


        if index in bm25_rank:

            score += 1.0 / (
                60 + bm25_rank[index]
            )


        if index in vector_rank:

            score += 1.0 / (
                60 + vector_rank[index]
            )


        hybrid_scores[index] = score


    top_hybrid = sorted(
        hybrid_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )[:20]


    # ========================================================
    # CROSS-ENCODER RERANKING
    # ========================================================

    candidate_indices = [
        index
        for index, _ in top_hybrid
    ]


    pairs = [

        [
            question,
            chunks[index]["text"]
        ]

        for index in candidate_indices
    ]


    reranker_scores = reranker.predict(
        pairs
    )


    reranked_results = [

        (
            candidate_indices[i],
            float(reranker_scores[i])
        )

        for i in range(
            len(candidate_indices)
        )
    ]


    reranked_results.sort(
        key=lambda item: item[1],
        reverse=True
    )


    # --------------------------------------------------------
    # Stop retrieval timer
    # --------------------------------------------------------

    retrieval_end = time.perf_counter()


    retrieval_time = (
        retrieval_end
        -
        retrieval_start
    )


    # ========================================================
    # TOP 5
    # ========================================================

    top_5 = [
        index
        for index, _ in reranked_results[:5]
    ]


    # ========================================================
    # FIND FIRST RELEVANT RANK
    # ========================================================

    first_relevant_rank = None


    for rank, index in enumerate(
        top_5,
        start=1
    ):

        if index in relevant_chunks:

            first_relevant_rank = rank

            break


    # ========================================================
    # RECALL@5
    # ========================================================

    if first_relevant_rank is not None:

        recall_at_5 = 1

    else:

        recall_at_5 = 0


    # ========================================================
    # MRR
    # ========================================================

    if first_relevant_rank is not None:

        reciprocal_rank = (
            1 / first_relevant_rank
        )

    else:

        reciprocal_rank = 0


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print("\nTop 5 retrieved chunks:")

    for rank, index in enumerate(
        top_5,
        start=1
    ):

        is_relevant = (
            index in relevant_chunks
        )

        print(
            f"Rank {rank}: "
            f"Chunk {index} "
            f"| Relevant: {is_relevant}"
        )


    print(
        "\nRecall@5:",
        recall_at_5
    )

    print(
        "First relevant rank:",
        first_relevant_rank
    )

    print(
        "Reciprocal Rank:",
        round(
            reciprocal_rank,
            4
        )
    )

    print(
        "Retrieval time:",
        round(
            retrieval_time,
            4
        ),
        "seconds"
    )


    # ========================================================
    # SAVE RESULT
    # ========================================================

    results.append({

        "question_number":
            question_number,

        "question":
            question,

        "expected_value":
            expected_value,

        "relevant_chunk_count":
            len(relevant_chunks),

        "first_relevant_rank":
            first_relevant_rank,

        "recall_at_5":
            recall_at_5,

        "reciprocal_rank":
            reciprocal_rank,

        "retrieval_time_seconds":
            retrieval_time
    })


# ============================================================
# 13. CLOSE QDRANT
# ============================================================

qdrant_client.close()


# ============================================================
# 14. CREATE DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# 15. CALCULATE FINAL METRICS
# ============================================================

recall_at_5_average = (
    results_df["recall_at_5"].mean()
)


mrr_average = (
    results_df["reciprocal_rank"].mean()
)


average_retrieval_time = (
    results_df[
        "retrieval_time_seconds"
    ].mean()
)


# ============================================================
# 16. DISPLAY FINAL RESULTS
# ============================================================

print("\n\n")
print("=" * 70)
print("FINAL RETRIEVAL EVALUATION")
print("=" * 70)


print(
    "\nRecall@5:",
    f"{recall_at_5_average * 100:.2f}%"
)


print(
    "MRR:",
    f"{mrr_average:.4f}"
)


print(
    "Average retrieval latency:",
    f"{average_retrieval_time:.4f} seconds"
)


print("\n")


# ============================================================
# 17. SAVE CSV
# ============================================================

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    "Detailed results saved to:"
)

print(
    OUTPUT_PATH
)


print("\nEvaluation completed.")