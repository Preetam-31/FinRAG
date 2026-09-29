import os
import json
import sys
import asyncio
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from openai import OpenAI
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from ragas import EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    ContextRecall,
    Faithfulness,
    FactualCorrectness,
)


# ============================================================
# 1. PATHS AND ENVIRONMENT
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

QUESTIONS_PATH = ROOT_DIR / "evaluation" / "questions.json"
PDF_PATH = ROOT_DIR / "data" / "documents" / "company_report.pdf"
CHECKPOINT_PATH = ROOT_DIR / "evaluation" / "checkpoint.json"
RESULTS_PATH = ROOT_DIR / "evaluation" / "results.csv"

load_dotenv(ROOT_DIR / ".env")

api_key = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:3b"
)

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434/v1"
)

RUN_RAGAS = os.getenv(
    "RUN_RAGAS",
    "false"
).lower() in ("1", "true", "yes")


# ============================================================
# 2. INITIALIZE MODELS
# ============================================================

# If the Gemini API key is missing, use Ollama directly.
gemini_disabled = not bool(api_key)

if gemini_disabled:
    print("Gemini API key missing.")
    print("Gemini disabled. Using Ollama.")

gemini_client = None

if api_key:
    gemini_client = genai.Client(api_key=api_key)

ollama_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
    timeout=300.0,
)


# ============================================================
# 3. OLLAMA ANSWER GENERATION
# ============================================================

def generate_with_ollama(prompt):
    """
    Generate an answer using local Ollama.
    """

    try:
        print(
            f"Answer generation model: Ollama - {OLLAMA_MODEL}"
        )

        response = ollama_client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
        )

        answer = response.choices[0].message.content

        if answer and answer.strip():
            print("Answer generated using Ollama.")

            return answer.strip(), "Ollama"

        return (
            "Ollama returned an empty answer.",
            "Ollama"
        )

    except Exception as error:

        print("Ollama request failed:", error)

        return (
            f"Answer generation failed. Ollama error: {error}",
            "None"
        )


# ============================================================
# 4. GEMINI ONCE, THEN OLLAMA
# ============================================================

def generate_answer(prompt):
    """
    Try Gemini only once per Python run.

    If Gemini fails, disable it for the rest of the run
    and use Ollama without retrying Gemini.
    """

    global gemini_disabled

    if not gemini_disabled:

        try:
            print("\nTrying Gemini (one attempt only)...")

            print(
                f"Answer generation model: Gemini - {GEMINI_MODEL}"
            )

            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )

            answer = getattr(response, "text", None)

            if answer and answer.strip():

                print("Answer generated using Gemini.")

                return answer.strip(), "Gemini"

            print("Gemini returned an empty answer.")

        except Exception as error:

            print("\nGemini request failed.")
            print("Error:", error)

        gemini_disabled = True

        print("Gemini disabled for the rest of this run.")

    print("\nSwitching to Ollama...")

    return generate_with_ollama(prompt)


# ============================================================
# 5. RULE-BASED FINANCIAL ANSWER EXTRACTION
# ============================================================

def extract_financial_answer(question, context):
    """
    Extract financial figures directly from retrieved context.

    The function selects the first figure in a financial
    statement row, assuming the table columns are ordered
    newest year to oldest year, such as 2025, 2024, 2023.

    Returns:
        Extracted answer if a matching figure is found.
        None if the requested figure is not found.
    """

    question_lower = question.lower()

    # Normalize whitespace while preserving line breaks.
    context = context.replace("\r", "\n")

    # --------------------------------------------------------
    # 5A. TOTAL NET SALES
    # --------------------------------------------------------

    if (
        "total net sales" in question_lower
        or "total revenue" in question_lower
        or "total sales" in question_lower
    ):

        patterns = [
            r"Total\s+net\s+sales\s*\$?\s*([\d,]+)",
            r"Total\s+revenue\s*\$?\s*([\d,]+)",
            r"Total\s+sales\s*\$?\s*([\d,]+)",
        ]

        for pattern in patterns:

            matches = re.findall(
                pattern,
                context,
                flags=re.IGNORECASE
            )

            if matches:

                # First value corresponds to the latest year
                # in the 2025, 2024, 2023 table.
                value = matches[0]

                return (
                    f"Total net sales were ${value} million "
                    f"for fiscal year 2025, according to "
                    f"the financial statement."
                )

    # --------------------------------------------------------
    # 5B. SERVICES NET SALES
    # --------------------------------------------------------

    if (
        "services" in question_lower
        and (
            "sales" in question_lower
            or "revenue" in question_lower
        )
    ):

        patterns = [
            r"Services\s*(?:\(\s*1\s*\))?\s*"
            r"\$?\s*([\d,]+)",

            r"Services\s+net\s+sales\s*\$?\s*([\d,]+)",
        ]

        for pattern in patterns:

            matches = re.findall(
                pattern,
                context,
                flags=re.IGNORECASE
            )

            if matches:

                value = matches[0]

                return (
                    f"Services net sales were ${value} million "
                    f"for fiscal year 2025."
                )

    # --------------------------------------------------------
    # 5C. NET INCOME
    # --------------------------------------------------------

    if "net income" in question_lower:

        patterns = [
            r"Net\s+income\s*\$?\s*([\d,]+)",
            r"Net\s+income\s*\(loss\)\s*\$?\s*([\d,]+)",
        ]

        for pattern in patterns:

            matches = re.findall(
                pattern,
                context,
                flags=re.IGNORECASE
            )

            if matches:

                value = matches[0]

                return (
                    f"Net income was ${value} million "
                    f"for fiscal year 2025."
                )

    # --------------------------------------------------------
    # 5D. RESEARCH AND DEVELOPMENT EXPENSE
    # --------------------------------------------------------

    if (
        "research" in question_lower
        or "development" in question_lower
        or "r&d" in question_lower
    ):

        patterns = [
            r"Research\s+and\s+development\s*"
            r"\$?\s*([\d,]+)",

            r"Research\s*&\s*development\s*"
            r"\$?\s*([\d,]+)",

            r"R\s*&\s*D\s*"
            r"\$?\s*([\d,]+)",
        ]

        for pattern in patterns:

            matches = re.findall(
                pattern,
                context,
                flags=re.IGNORECASE
            )

            if matches:

                value = matches[0]

                return (
                    f"Research and development expense was "
                    f"${value} million for fiscal year 2025."
                )

    # --------------------------------------------------------
    # 5E. TOTAL OPERATING EXPENSES
    # --------------------------------------------------------

    if (
        "operating expenses" in question_lower
        or "total operating expense" in question_lower
    ):

        patterns = [
            r"Total\s+operating\s+expenses\s*"
            r"\$?\s*([\d,]+)",

            r"Total\s+operating\s+expense\s*"
            r"\$?\s*([\d,]+)",
        ]

        for pattern in patterns:

            matches = re.findall(
                pattern,
                context,
                flags=re.IGNORECASE
            )

            if matches:

                value = matches[0]

                return (
                    f"Total operating expenses were "
                    f"${value} million for fiscal year 2025."
                )

    # --------------------------------------------------------
    # 5F. NO EXACT MATCH
    # --------------------------------------------------------

    return None


# ============================================================
# 6. LOAD QUESTIONS AND CHECKPOINT
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


if not isinstance(questions, list) or not questions:

    raise SystemExit(
        "questions.json must contain a non-empty JSON list."
    )


print(
    "\nNumber of evaluation questions:",
    len(questions)
)


for item in questions:

    if "question" not in item or "reference" not in item:

        raise SystemExit(
            "Every item in questions.json must contain "
            "'question' and 'reference'."
        )


# ------------------------------------------------------------
# Load existing checkpoint
# ------------------------------------------------------------

if CHECKPOINT_PATH.exists():

    try:

        with CHECKPOINT_PATH.open(
            "r",
            encoding="utf-8"
        ) as file:

            evaluation_data = json.load(file)

        if not isinstance(evaluation_data, list):
            evaluation_data = []

    except (json.JSONDecodeError, OSError):

        print(
            "Checkpoint could not be read; "
            "starting a new checkpoint."
        )

        evaluation_data = []

else:

    evaluation_data = []


valid_questions = {
    item["question"]
    for item in questions
}

checkpoint_by_question = {}


for item in evaluation_data:

    if (
        isinstance(item, dict)
        and item.get("user_input") in valid_questions
        and item.get("response")
        and item.get("retrieved_contexts")
    ):

        checkpoint_by_question[
            item["user_input"]
        ] = item


evaluation_data = list(
    checkpoint_by_question.values()
)

completed_questions = set(
    checkpoint_by_question.keys()
)


print(
    "Completed questions loaded from checkpoint:",
    len(completed_questions)
)

print(
    "Checkpoint:",
    CHECKPOINT_PATH
)


# ------------------------------------------------------------
# RESET OLD CHECKPOINT FOR A FRESH EVALUATION RUN
# ------------------------------------------------------------

print("\nStarting a fresh answer-generation run.")

evaluation_data = []

checkpoint_by_question = {}

completed_questions = set()

print("Old checkpoint entries cleared from memory.")
print("All evaluation questions will be regenerated.")



# ------------------------------------------------------------
# Save checkpoint safely
# ------------------------------------------------------------

def save_checkpoint(data):
    """
    Save checkpoint using a temporary file.
    """

    temporary_path = CHECKPOINT_PATH.with_suffix(".tmp")

    with temporary_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )

    temporary_path.replace(CHECKPOINT_PATH)


# ============================================================
# 7. LOAD PDF AND CREATE RETRIEVAL INDEXES
# ============================================================

sys.path.insert(
    0,
    str(ROOT_DIR / "src")
)

from ingestion import (
    extract_text_from_pdf,
    create_chunks
)


if not PDF_PATH.exists():

    raise SystemExit(
        f"PDF file not found: {PDF_PATH}"
    )


pages = extract_text_from_pdf(
    str(PDF_PATH)
)

chunks = create_chunks(pages)


if not chunks:

    raise SystemExit(
        "No document chunks were created from the PDF."
    )


print(
    "Number of document chunks:",
    len(chunks)
)


# ------------------------------------------------------------
# Create BM25 index
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Load embedding model
# ------------------------------------------------------------

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ------------------------------------------------------------
# Load cross-encoder reranker
# ------------------------------------------------------------

print("Loading Cross-Encoder...")

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)


# ============================================================
# 8. CONNECT TO QDRANT
# ============================================================

qdrant_client = QdrantClient(
    path=str(ROOT_DIR / "qdrant_data")
)

collection_name = "finrag_documents"


# ============================================================
# 9. RUN FINRAG
# ============================================================

try:

    for question_number, question_data in enumerate(
        questions,
        start=1
    ):

        query = question_data["question"]
        reference = question_data["reference"]

        # ----------------------------------------------------
        # Skip completed questions
        # ----------------------------------------------------

        if query in completed_questions:

            print(
                f"\nSkipping question {question_number}; "
                "already in checkpoint."
            )

            continue


        print("\n" + "=" * 70)

        print(
            f"EVALUATION QUESTION {question_number} "
            f"of {len(questions)}"
        )

        print("=" * 70)

        print("Question:", query)


        # ====================================================
        # 9A. BM25 RETRIEVAL
        # ====================================================

        query_tokens = query.lower().split()

        bm25_scores = bm25.get_scores(
            query_tokens
        )

        bm25_top_indices = (
            bm25_scores.argsort()[-20:][::-1]
        )


        # ====================================================
        # 9B. VECTOR RETRIEVAL
        # ====================================================

        query_embedding = embedding_model.encode(
            query
        ).tolist()

        vector_results = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=10,
            with_payload=True,
        ).points


        print(
            "Vector retrieval returned",
            len(vector_results),
            "candidates."
        )


        # ====================================================
        # 9C. RANK BM25 RESULTS
        # ====================================================

        bm25_rank = {
            int(index): rank + 1
            for rank, index in enumerate(
                bm25_top_indices
            )
        }


        # ====================================================
        # 9D. RANK VECTOR RESULTS
        # ====================================================

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


        # ====================================================
        # 9E. RECIPROCAL RANK FUSION
        # ====================================================

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
            reverse=True,
        )[:20]


        if not top_hybrid:

            print(
                "No retrieval results found; "
                "stopping without scoring."
            )

            break


        # ====================================================
        # 9F. CROSS-ENCODER RERANKING
        # ====================================================

        candidate_indices = [
            index
            for index, _ in top_hybrid
        ]

        pairs = [
            [
                query,
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


        # ====================================================
        # 9G. SELECT TOP EIGHT CONTEXTS
        # ====================================================

        top_results = reranked_results[:8]

        contexts = [
            chunks[index]["text"]
            for index, _ in top_results
        ]


        # ====================================================
        # 9H. BUILD SOURCE-LABELLED CONTEXT
        # ====================================================

        context_parts = []

        for position, (index, _) in enumerate(
            top_results,
            start=1
        ):

            context_parts.append(
                f"Source {position}\n"
                f"Page: {chunks[index].get('page_number', 'Not available')}\n"
                f"Text: {chunks[index]['text']}\n"
            )


        context_text = "\n".join(
            context_parts
        )


        # ----------------------------------------------------
        # Print retrieved context for debugging
        # ----------------------------------------------------

        print("\n" + "=" * 70)

        print(
            "RETRIEVED CONTEXT SENT TO THE ANSWER MODEL"
        )

        print("=" * 70)

        print(context_text)

        print("=" * 70)


        # ====================================================
        # 9I. BUILD PROMPT
        # ====================================================

        prompt = f"""
You are a financial question-answering assistant.

Answer the question using ONLY the information
provided in the context.

The context contains financial tables.
Numbers may be listed in columns for 2025, 2024,
and 2023.

For questions about 2025, use the 2025 value.

IMPORTANT:
- Read the financial tables carefully.
- Do not say the information is missing if the
  exact value is present in the context.
- Do not confuse 2025 values with 2024 or 2023.
- Express financial amounts in millions of dollars.
- Answer in one complete sentence.
- Include the financial item, year, currency, and unit.
- Do not add unsupported explanations.

Examples:
Question: What was Apple's total net sales in 2025?
Answer: Apple's total net sales in 2025 were $416,161 million.

Question: What was Apple's net income in 2025?
Answer: Apple's net income in 2025 was $112,010 million.

Question:
{query}

Document passages:
{context_text}

Answer:
"""


        # ====================================================
       # ====================================================
# 9J. LLM GENERATION FIRST
# ====================================================

# Bypass rule-based extraction for this evaluation.
# Send the question and retrieved context to Gemini,
# with Ollama as fallback.

        answer, model_used = generate_answer(
            prompt
)

        print(
               f"\nAnswer generated using: {model_used}"
)


        # ====================================================
        # 9K. DISPLAY ANSWER
        # ====================================================

        print("\nGenerated Answer:")

        print(answer)

        print("Model used:", model_used)


        # ====================================================
        # 9L. SAVE CHECKPOINT
        # ====================================================

        record = {
            "user_input": query,
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": reference,
            "model_used": model_used,
        }


        evaluation_data.append(
            record
        )

        completed_questions.add(
            query
        )


        save_checkpoint(
            evaluation_data
        )

        print("Checkpoint saved.")


        # ====================================================
        # 9M. UPDATE ANSWER CSV
        # ====================================================

        answer_rows = []

        for saved in evaluation_data:

            answer_rows.append(
                {
                    "question": saved.get(
                        "user_input",
                        ""
                    ),

                    "answer": saved.get(
                        "response",
                        ""
                    ),

                    "reference": saved.get(
                        "reference",
                        ""
                    ),

                    "model_used": saved.get(
                        "model_used",
                        "Unknown"
                    ),
                }
            )


        pd.DataFrame(
            answer_rows
        ).to_csv(
            RESULTS_PATH,
            index=False
        )


finally:

    qdrant_client.close()


# ============================================================
# 10. CHECK WHETHER ALL QUESTIONS ARE COMPLETE
# ============================================================

missing_questions = [
    item["question"]
    for item in questions
    if item["question"] not in completed_questions
]


if missing_questions:

    print("\n" + "=" * 70)

    print(
        "EVALUATION PAUSED - NOT ALL QUESTIONS ARE COMPLETE"
    )

    print("=" * 70)

    print(
        f"Completed: {len(completed_questions)} "
        f"/ {len(questions)}"
    )

    print(
        "The successful answers are saved in:",
        CHECKPOINT_PATH
    )

    print(
        "Run this script again later to resume."
    )

    print("Questions still pending:")

    for query in missing_questions:

        print("-", query)

    raise SystemExit(0)


# ============================================================
# 11. RESTORE QUESTION ORDER
# ============================================================

evaluation_data = [
    checkpoint_by_question[question["question"]]
    if question["question"] in checkpoint_by_question
    else next(
        item
        for item in evaluation_data
        if item["user_input"] == question["question"]
    )
    for question in questions
]


save_checkpoint(
    evaluation_data
)


# ============================================================
# ============================================================
# 12. OPTIONAL RAGAS EVALUATION
# ============================================================

if not RUN_RAGAS:
    print("\nRUN_RAGAS is disabled.")
    print("Answers are saved in:", RESULTS_PATH)
    print(
        "To run RAGAS later, set RUN_RAGAS=true in .env "
        "and rerun."
    )
    raise SystemExit(0)


# ============================================================
# 13. CREATE RAGAS EVALUATION DATASET
# ============================================================

print("\n" + "=" * 70)
print("CREATING RAGAS EVALUATION DATASET")
print("=" * 70)

evaluation_samples = [
    SingleTurnSample(
        user_input=item["user_input"],
        retrieved_contexts=item["retrieved_contexts"],
        response=item["response"],
        reference=item["reference"],
    )
    for item in evaluation_data
]

evaluation_dataset = EvaluationDataset(
    samples=evaluation_samples
)

print("Evaluation samples:", len(evaluation_samples))


# ============================================================
# 14. CREATE RAGAS EVALUATOR WITH GEMINI / OLLAMA FALLBACK
# ============================================================

from openai import AsyncOpenAI
from ragas.llms import llm_factory

print("\nCreating RAGAS evaluator...")


# Model used for Gemini evaluation
RAGAS_MODEL = os.getenv(
    "RAGAS_MODEL",
    GEMINI_MODEL
)

# Track which evaluator is currently active
ragas_backend = "gemini" if api_key else "ollama"

# Global metric objects
context_recall_metric = None
faithfulness_metric = None
factual_correctness_metric = None


def create_ragas_metrics(backend):
    """
    Create RAGAS metrics using either Gemini or Ollama.

    Gemini uses Google's OpenAI-compatible endpoint.
    Ollama uses its local OpenAI-compatible endpoint.
    """

    print("\nInitializing RAGAS with:", backend.upper())

    if backend == "gemini":

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing."
            )

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=(
                "https://generativelanguage.googleapis.com/"
                "v1beta/openai/"
            ),
            timeout=120.0,
            max_retries=0,
        )

        model_name = RAGAS_MODEL

    else:

        # Local Ollama evaluator
        client = AsyncOpenAI(
            api_key="ollama",
            base_url=OLLAMA_BASE_URL,
            timeout=300.0,
            max_retries=0,
        )

        model_name = OLLAMA_MODEL

    evaluator_llm = llm_factory(
        model_name,
        provider="openai",
        client=client,
        temperature=0,
    )

    recall_metric = ContextRecall(
        llm=evaluator_llm
    )

    faithfulness = Faithfulness(
        llm=evaluator_llm
    )

    factual_correctness = FactualCorrectness(
        llm=evaluator_llm
    )

    return (
        recall_metric,
        faithfulness,
        factual_correctness,
    )


def initialize_ragas(backend):
    """
    Initialize or replace all three RAGAS metrics.
    """

    global ragas_backend
    global context_recall_metric
    global faithfulness_metric
    global factual_correctness_metric

    (
        context_recall_metric,
        faithfulness_metric,
        factual_correctness_metric,
    ) = create_ragas_metrics(backend)

    ragas_backend = backend

    print(
        "RAGAS evaluator ready:",
        backend.upper()
    )


# Use Gemini initially if an API key is available.
# Otherwise, start directly with Ollama.

try:

    initialize_ragas(ragas_backend)

except Exception as error:

    print("\nGemini RAGAS initialization failed.")
    print("Error:", error)

    print("\nSwitching RAGAS to local Ollama...")

    initialize_ragas("ollama")


# ============================================================
# 15. SCORE EACH SAMPLE
# ============================================================

async def score_sample_once(sample):
    """
    Score one sample using the currently active evaluator.
    """

    # ----------------------------------------
    # Context Recall
    # ----------------------------------------

    recall_result = await context_recall_metric.ascore(
        user_input=sample.user_input,
        retrieved_contexts=sample.retrieved_contexts,
        reference=sample.reference,
    )

    # ----------------------------------------
    # Faithfulness
    # ----------------------------------------

    faithfulness_result = await faithfulness_metric.ascore(
        user_input=sample.user_input,
        response=sample.response,
        retrieved_contexts=sample.retrieved_contexts,
    )

    # ----------------------------------------
    # Factual Correctness
    # ----------------------------------------

    factual_result = await factual_correctness_metric.ascore(
        
        response=sample.response,
        reference=sample.reference,
    )

    return {
        "context_recall": float(
            recall_result.value
        ),
        "faithfulness": float(
            faithfulness_result.value
        ),
        "factual_correctness": float(
            factual_result.value
        ),
    }


async def score_sample(sample):
    """
    First try the current RAGAS evaluator.

    If Gemini fails, switch to Ollama and retry
    the complete sample using Ollama.

    Once switched, Ollama remains active for all
    remaining samples.
    """

    global ragas_backend

    try:

        return await score_sample_once(sample)

    except Exception as error:

        # Do not attempt another Gemini request
        # after an evaluation failure.

        if ragas_backend != "gemini":
            print(
                "\nOllama RAGAS evaluation failed."
            )
            raise

        print("\nGemini RAGAS evaluation failed.")
        print("Error:", error)

        print(
            "\nSwitching RAGAS evaluator "
            "to local Ollama..."
        )

        # Replace all metrics with local Ollama metrics.
        initialize_ragas("ollama")

        print(
            "\nRetrying the current sample "
            "using Ollama..."
        )

        # Retry the whole sample using Ollama.
        # No further Gemini calls will be made.
        result = await score_sample_once(sample)

        print(
            "Current sample evaluated using Ollama."
        )

        return result


# ============================================================
# 16. RUN RAGAS EVALUATION
# ============================================================

async def run_ragas_evaluation():

    scored = []

    for number, sample in enumerate(
        evaluation_dataset.samples,
        start=1
    ):

        print("\n" + "-" * 70)

        print(
            f"Evaluating sample {number} "
            f"of {len(evaluation_dataset.samples)}"
        )

        print(
            "Question:",
            sample.user_input
        )

        print(
            "RAGAS evaluator:",
            ragas_backend.upper()
        )

        result = await score_sample(sample)

        print(
            "Context Recall:",
            round(
                result["context_recall"],
                4
            )
        )

        print(
            "Faithfulness:",
            round(
                result["faithfulness"],
                4
            )
        )

        print(
            "Factual Correctness:",
            round(
                result["factual_correctness"],
                4
            )
        )

        scored.append(result)

    return scored


# ============================================================
# 17. EXECUTE EVALUATION
# ============================================================

metric_results = asyncio.run(
    run_ragas_evaluation()
)


# ============================================================
# 18. DISPLAY AND SAVE FINAL RESULTS
# ============================================================

results_df = pd.DataFrame(
    metric_results
)

results_df.insert(
    0,
    "question",
    [
        item["user_input"]
        for item in evaluation_data
    ]
)

print("\nDETAILED FINRAG EVALUATION RESULTS")

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# AVERAGE SCORES
# ============================================================

print("\nAVERAGE SCORES")

for metric_name in (
    "context_recall",
    "faithfulness",
    "factual_correctness"
):

    print(
        f"{metric_name.replace('_', ' ').title()}: "
        f"{results_df[metric_name].mean():.4f}"
    )


# ============================================================
# SAVE RAGAS RESULTS SEPARATELY
# ============================================================

RAGAS_RESULTS_PATH = (
    ROOT_DIR
    / "evaluation"
    / "ragas_results.csv"
)

results_df.to_csv(
    RAGAS_RESULTS_PATH,
    index=False
)

print(
    "\nRAGAS results saved to:",
    RAGAS_RESULTS_PATH
)

print(
    "Original answer results preserved at:",
    RESULTS_PATH
)

print(
    "Final RAGAS evaluator used:",
    ragas_backend.upper()
)

print(
    "\nFinRAG evaluation completed successfully."
)