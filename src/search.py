from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

client = QdrantClient(
    path="qdrant_data"
)

collection_name = "finrag_documents"


query = input("\nEnter your question: ")

query_embedding = model.encode(query).tolist()


results = client.query_points(
    collection_name=collection_name,
    query=query_embedding,
    limit=5,
    with_payload=True
).points


print("\nTop relevant sections:\n")


for i, result in enumerate(results):

    print("=" * 70)

    print("Result:", i + 1)

    print("Page:", result.payload["page_number"])

    print("\nSimilarity Score:")
    print(result.score)

    print("\nText:")
    print(result.payload["text"])


client.close()