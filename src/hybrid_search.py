from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from ingestion import extract_text_from_pdf, create_chunks


# --------------------------------------------------
# 1. Load document and create chunks
# --------------------------------------------------

pdf_path = "data/documents/company_report.pdf"

pages = extract_text_from_pdf(pdf_path)

chunks = create_chunks(pages)

print("Number of chunks:", len(chunks))


# --------------------------------------------------
# 2. Create BM25 index
# --------------------------------------------------

documents = []

for chunk in chunks:
    documents.append(chunk["text"])


tokenized_documents = []

for document in documents:
    tokens = document.lower().split()
    tokenized_documents.append(tokens)


bm25 = BM25Okapi(tokenized_documents)


# --------------------------------------------------
# 3. Load embedding model
# --------------------------------------------------

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 4. Connect to Qdrant
# --------------------------------------------------

qdrant_client = QdrantClient(
    path="qdrant_data"
)

collection_name = "finrag_documents"


# --------------------------------------------------
# 5. Ask question
# --------------------------------------------------

query = input("\nEnter your question: ")


# --------------------------------------------------
# 6. BM25 search
# --------------------------------------------------

query_tokens = query.lower().split()

bm25_scores = bm25.get_scores(query_tokens)

bm25_top_indices = bm25_scores.argsort()[-10:][::-1]


# --------------------------------------------------
# 7. Vector search
# --------------------------------------------------

query_embedding = embedding_model.encode(
    query
).tolist()

vector_results = qdrant_client.query_points(
    collection_name=collection_name,
    query=query_embedding,
    limit=10,
    with_payload=True
).points


# --------------------------------------------------
# 8. Create rankings
# --------------------------------------------------

bm25_rank = {}

for rank, index in enumerate(bm25_top_indices):
    bm25_rank[index] = rank + 1


vector_rank = {}

for rank, result in enumerate(vector_results):
    vector_rank[result.id] = rank + 1


# --------------------------------------------------
# 9. Hybrid scoring using RRF
# --------------------------------------------------

hybrid_scores = {}

all_indices = set(bm25_rank.keys()) | set(vector_rank.keys())

for index in all_indices:

    score = 0

    if index in bm25_rank:
        score += 1 / (60 + bm25_rank[index])

    if index in vector_rank:
        score += 1 / (60 + vector_rank[index])

    hybrid_scores[index] = score


# --------------------------------------------------
# 10. Sort hybrid results
# --------------------------------------------------

top_hybrid = sorted(
    hybrid_scores.items(),
    key=lambda x: x[1],
    reverse=True
)[:5]


# --------------------------------------------------
# 11. Display results
# --------------------------------------------------

print("\n" + "=" * 70)
print("HYBRID SEARCH RESULTS")
print("=" * 70)


for position, (index, score) in enumerate(top_hybrid):

    print("\n" + "-" * 70)

    print("Result:", position + 1)

    print("Page:", chunks[index]["page_number"])

    print("Hybrid Score:", round(score, 5))

    if index in bm25_rank:
        print("BM25 Rank:", bm25_rank[index])
    else:
        print("BM25 Rank: Not in top 10")

    if index in vector_rank:
        print("Vector Rank:", vector_rank[index])
    else:
        print("Vector Rank: Not in top 10")

    print("\nText:")

    print(chunks[index]["text"])


qdrant_client.close()