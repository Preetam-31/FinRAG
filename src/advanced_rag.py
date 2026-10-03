
import os
import json
import urllib.request
import urllib.error
from pathlib import Path

from dotenv import load_dotenv
import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer, CrossEncoder

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from rank_bm25 import BM25Okapi

from groq import Groq


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print("Groq client initialized.")
else:
    groq_client = None
    print("Groq API key not found. Ollama will be used.")

GROQ_DISABLED = False


# ============================================================
# 2. CONFIGURATION
# ============================================================
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT_DIR / "data" / "documents" / "company_report.pdf"


COLLECTION_NAME = "finrag_documents"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RERANKER_MODEL = "BAAI/bge-reranker-base"

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)

OLLAMA_MODEL = "qwen2.5:3b"

OLLAMA_URL = "http://ollama:11434/api/generate"


# ============================================================
# 3. LOAD PDF
# ============================================================

def extract_text_from_pdf(pdf_path):

    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document):

        text = page.get_text()

        if text.strip():

            pages.append({
                "page": page_number + 1,
                "text": text
            })

    document.close()

    return pages


# ============================================================
# 4. CREATE CHUNKS
# ============================================================

def create_chunks(pages):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = []

    for page in pages:

        page_chunks = text_splitter.split_text(
            page["text"]
        )

        for chunk in page_chunks:

            chunks.append({
                "text": chunk,
                "page": page["page"]
            })

    return chunks


# ============================================================
# 5. LOAD DOCUMENT
# ============================================================

print("Loading financial document...")

pages = extract_text_from_pdf(PDF_PATH)

chunks = create_chunks(pages)

print(f"Number of chunks: {len(chunks)}")


# ============================================================
# 6. LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


# ============================================================
# 7. CREATE EMBEDDINGS
# ============================================================

texts = []

for chunk in chunks:
    texts.append(chunk["text"])


embeddings = embedding_model.encode(
    texts,
    show_progress_bar=True
)


# ============================================================
# 8. QDRANT VECTOR DATABASE
# ============================================================

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

collection_name = COLLECTION_NAME


# ============================================================
# 9. CREATE QDRANT COLLECTION
# ============================================================

existing_collections = qdrant_client.get_collections()

collection_exists = False

for collection in existing_collections.collections:

    if collection.name == collection_name:

        collection_exists = True
        break


if not collection_exists:

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )

    print("Qdrant collection created.")


# ============================================================
# 10. INSERT VECTORS
# ============================================================

points = []

for index, embedding in enumerate(embeddings):

    points.append(
        PointStruct(
            id=index,
            vector=embedding.tolist(),
            payload={
                "text": chunks[index]["text"],
                "page": chunks[index]["page"]
            }
        )
    )


try:

    collection_info = qdrant_client.get_collection(
        collection_name
    )

    existing_count = collection_info.points_count

except Exception:

    existing_count = 0


if existing_count == 0:

    qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )

    print("Vectors inserted into Qdrant.")

else:

    print(
        f"Qdrant already contains {existing_count} points."
    )


# ============================================================
# 11. CREATE BM25 INDEX
# ============================================================

tokenized_documents = []

for text in texts:

    tokenized_documents.append(
        text.lower().split()
    )


bm25 = BM25Okapi(
    tokenized_documents
)


# ============================================================
# 12. LOAD CROSS ENCODER
# ============================================================

print("Cross-Encoder disabled for Docker deployment.")

reranker = None


# ============================================================
# 13. VECTOR SEARCH
# ============================================================

def vector_search(query, top_k=10):

    query_embedding = embedding_model.encode(
        query
    )

    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embedding.tolist(),
        limit=top_k
    ).points

    vector_results = []

    for result in results:

        vector_results.append({
            "text": result.payload["text"],
            "page": result.payload["page"],
            "score": result.score
        })

    return vector_results


# ============================================================
# 14. BM25 SEARCH
# ============================================================

def bm25_search(query, top_k=10):

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(
        tokenized_query
    )

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True
    )

    results = []

    for index in ranked_indexes[:top_k]:

        results.append({
            "text": chunks[index]["text"],
            "page": chunks[index]["page"],
            "score": float(scores[index])
        })

    return results


# ============================================================
# 15. HYBRID SEARCH USING RRF
# ============================================================

def hybrid_search(query, top_k=10):

    bm25_results = bm25_search(
        query,
        top_k=10
    )

    vector_results = vector_search(
        query,
        top_k=10
    )

    combined_results = {}

    # BM25 RRF

    for rank, result in enumerate(bm25_results):

        key = result["text"]

        if key not in combined_results:

            combined_results[key] = {
                "text": result["text"],
                "page": result["page"],
                "score": 0
            }

        combined_results[key]["score"] += (
            1 / (60 + rank + 1)
        )

    # Vector RRF

    for rank, result in enumerate(vector_results):

        key = result["text"]

        if key not in combined_results:

            combined_results[key] = {
                "text": result["text"],
                "page": result["page"],
                "score": 0
            }

        combined_results[key]["score"] += (
            1 / (60 + rank + 1)
        )

    # Sort by RRF score

    results = list(
        combined_results.values()
    )

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


# ============================================================
# 16. CROSS-ENCODER RERANKING
# ============================================================

def rerank_results(query, results, top_k=5):

    if not results:
        return []

    pairs = []

    for result in results:
        pairs.append([
            query,
            result["text"]
        ])

    if reranker is not None:
        scores = reranker.predict(
            pairs
        )
    else:
        scores = [0.0] * len(results)

    reranked_results = []

    for index, score in enumerate(scores):
        reranked_results.append({
            "text": results[index]["text"],
            "page": results[index]["page"],
            "score": float(score)
        })

    reranked_results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return reranked_results[:top_k]


# ============================================================
# 17. GEMINI + OLLAMA GENERATION WITH FALLBACK
# ============================================================

def generate_answer(prompt):
    global GROQ_DISABLED

    # Try Groq first
    if groq_client is not None and not GROQ_DISABLED:
        try:
            print("Trying Groq...")

            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                max_tokens=200
            )

            answer = response.choices[0].message.content

            if answer and answer.strip():
                answer = answer.strip()

                prefixes = [
                    "Answer:",
                    "Answer -",
                    "Response:",
                    "Final answer:"
                ]

                for prefix in prefixes:
                    if answer.lower().startswith(prefix.lower()):
                        answer = answer[len(prefix):].strip()

                print("Groq answered successfully.")
                print("Generated answer:", answer)

                return answer

            print("Groq returned an empty response.")

        except Exception as e:
            print("Groq failed:")
            print(str(e))

            GROQ_DISABLED = True

            print("Switching to Ollama.")

    # Ollama fallback
    try:
        print("Generating answer using Ollama...")

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 200
            }
        }

        data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            OLLAMA_URL,
            data=data,
            headers={
                "Content-Type": "application/json"
            },
            method="POST"
        )

        with urllib.request.urlopen(
            request,
            timeout=180
        ) as response:

            result = json.loads(
                response.read().decode("utf-8")
            )

        answer = result.get(
            "response",
            ""
        ).strip()

        if not answer:
            raise Exception(
                "Ollama returned an empty response."
            )

        prefixes = [
            "Answer:",
            "Answer -",
            "Response:",
            "Final answer:"
        ]

        for prefix in prefixes:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()

        print("Ollama answered successfully.")
        print("Generated answer:", answer)

        return answer

    except Exception as e:
        print("Ollama failed:")
        print(str(e))

        raise Exception(
            "Ollama failed. Please check that the Ollama server is running and the model is available."
        )


# ============================================================
# 18. CLEAN SOURCE TEXT FOR UI
# ============================================================

def clean_source_text(text):

    text = " ".join(text.split())

    text = text.replace(" ,", ",")
    text = text.replace(" .", ".")
    text = text.replace(" :", ":")
    text = text.replace(" ;", ";")

    return text.strip()


# ============================================================
# 19. EXTRACT QUERY-RELEVANT EVIDENCE
# ============================================================

def extract_relevant_evidence(
    text,
    query,
    max_chars=220
):

    clean_text = " ".join(
        text.split()
    )

    query_words = query.lower().split()

    stop_words = {
        "what",
        "was",
        "were",
        "the",
        "is",
        "in",
        "of",
        "for",
        "and",
        "a",
        "an",
        "how",
        "much",
        "did",
        "apple",
        "2025"
    }

    keywords = []

    for word in query_words:

        word = word.strip("?,.!")

        if word not in stop_words and len(word) > 2:

            keywords.append(word)

    lower_text = clean_text.lower()

    best_position = -1
    best_score = 0

    for keyword in keywords:

        position = lower_text.find(keyword)

        if position != -1:

            score = 1

            if (
                "net income" in lower_text
                and keyword == "income"
            ):

                score += 2

            if (
                "net sales" in lower_text
                and keyword == "sales"
            ):

                score += 2

            if (
                "operating expenses" in lower_text
                and keyword == "expenses"
            ):

                score += 2

            if (
                "research" in lower_text
                and keyword == "research"
            ):

                score += 2

            if score > best_score:

                best_score = score
                best_position = position

    # If no keyword was found

    if best_position == -1:

        return clean_text[:max_chars]

    start = max(
        0,
        best_position - 40
    )

    end = min(
        len(clean_text),
        best_position + 180
    )

    evidence = clean_text[start:end]

    if start > 0:

        evidence = "... " + evidence

    if end < len(clean_text):

        evidence = evidence + " ..."

    return evidence


# ============================================================
# 20. MAIN RAG FUNCTION
# ============================================================

def answer_question(query):

    # --------------------------------------------------------
    # VALIDATE QUERY
    # --------------------------------------------------------

    if not query or not query.strip():

        return {
            "answer": "Please enter a valid question.",
            "sources": []
        }

    query = query.strip()

    # --------------------------------------------------------
    # HYBRID RETRIEVAL
    # --------------------------------------------------------

    hybrid_results = hybrid_search(
        query,
        top_k=10
    )

    if not hybrid_results:

        return {
            "answer": (
                "I could not find relevant information "
                "in the document."
            ),
            "sources": []
        }

    # --------------------------------------------------------
    # CROSS-ENCODER RERANKING
    # --------------------------------------------------------

    reranked_results = rerank_results(
        query,
        hybrid_results,
        top_k=5
    )

    if not reranked_results:

        return {
            "answer": (
                "I could not find relevant information "
                "in the document."
            ),
            "sources": []
        }

    # --------------------------------------------------------
    # CREATE CONTEXT
    # --------------------------------------------------------

    context_parts = []

    for result in reranked_results:

        context_parts.append(
            f"Page {result['page']}:\n"
            f"{result['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # CREATE IMPROVED PROMPT
    # --------------------------------------------------------

    prompt = f"""
Answer the question using ONLY the financial document context below.

Question:
{query}

Context:
{context}

Rules:
- Give exactly one short sentence.
- Give the exact value requested.
- Use the correct year.
- Do not copy the table.
- Do not mention the context.
- Do not explain your reasoning.
- Do not write "Answer:".
- Do not repeat the question.
- Do not invent information.
- If the value cannot be found, say:
I could not find this information in the document.

Example:
Question: What was Apple's total net sales in 2025?
Correct answer: Total net sales were $416,161 million.

Now answer the question.


QUESTION:
{query}

DOCUMENT CONTEXT:
{context}

FOLLOW THESE RULES STRICTLY:

1. Answer only the exact question asked.

2. Identify the exact financial item requested.
   For example:
   - Net sales means net sales, not operating income.
   - Net income means net income, not operating income.
   - Research and development expense means R&D expense.
   - Total operating expenses means total operating expenses,
     not cost of sales or total costs.

3. If the question asks about a specific year, use only
   the value for that year.

4. Read the financial statement row and its corresponding
   value carefully. Do not take a number from a nearby row.

5. Financial tables may contain multiple years and many
   amounts. Do not list unrelated values from the table.

6. Do not copy the entire table or repeat the context.

7. Do not include calculations unless the question
   explicitly asks for them.

8. Do not invent numbers, units, or financial facts.

9. If the requested value is not clearly available in
   the context, respond exactly:
   I could not find this information in the document.

10. Do not include source page numbers, citations,
    explanations about the context, or introductory text.

11. Keep the answer to one short sentence.

12. If the question asks for a monetary value, include
    the currency and unit only when supported by the context.

EXAMPLES OF THE REQUIRED ANSWER STYLE:

Question: What were the total net sales in 2025?
Answer: Total net sales were $416,161 million.

Question: What was the net income in 2025?
Answer: Net income was $112,010 million.

Question: What was the R&D expense in 2025?
Answer: Research and development expense was $34,550 million.

These examples demonstrate the answer format only.
Always use the actual value from the document context.

Now answer the user's question.

Answer:
"""

    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    answer = generate_answer(
        prompt
    )

    # --------------------------------------------------------
    # CLEAN GENERATED ANSWER
    # --------------------------------------------------------

    answer = answer.strip()

    # Remove common introductory labels if generated.

    prefixes_to_remove = [
        "Answer:",
        "Answer -",
        "Answer:"
    ]

    for prefix in prefixes_to_remove:

        if answer.lower().startswith(prefix.lower()):

            answer = answer[len(prefix):].strip()

    # --------------------------------------------------------
    # CREATE SOURCES
    # --------------------------------------------------------

    sources = []

    for result in reranked_results:

        cleaned_text = clean_source_text(
            result["text"]
        )

        relevant_evidence = extract_relevant_evidence(
            cleaned_text,
            query
        )

        sources.append({
            "page": result["page"],
            "score": result["score"],
            "text": cleaned_text,
            "evidence": relevant_evidence
        })

    # --------------------------------------------------------
    # RETURN FINAL RESULT
    # --------------------------------------------------------

    return {
        "answer": answer,
        "sources": sources
    }


# ============================================================
# 21. OPTIONAL TERMINAL TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("======================================")
    print("        FinRAG Terminal Test")
    print("======================================")
    print()

    user_query = input(
        "Enter your question: "
    )

    result = answer_question(
        user_query
    )

    print()
    print("======================================")
    print("ANSWER")
    print("======================================")

    print(result["answer"])

    print()
    print("======================================")
    print("SOURCES")
    print("======================================")

    for source in result["sources"]:

        print(
            f"\nPage: {source['page']}"
        )

        print(
            f"Reranker Score: {source['score']}"
        )

        print(
            "Evidence:",
            source["evidence"]
        )

        print(
            "Source text:",
            source["text"][:500]
        )

