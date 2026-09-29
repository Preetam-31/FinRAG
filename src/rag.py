import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from google import genai

# Load environment variables
load_dotenv()

# Load embedding model
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Connect to Qdrant
qdrant_client = QdrantClient(path="qdrant_data")

collection_name = "finrag_documents"

# Connect to Gemini
gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Ask user a question
query = input("\nAsk a question about the document: ")

# Convert question into embedding
query_embedding = embedding_model.encode(query).tolist()

# Retrieve relevant chunks
results = qdrant_client.query_points(
    collection_name=collection_name,
    query=query_embedding,
    limit=5,
    with_payload=True
).points

# Build context
context = ""

for i, result in enumerate(results):
    context += (
        f"\nSource {i + 1}\n"
        f"Page: {result.payload['page_number']}\n"
        f"Text: {result.payload['text']}\n"
    )

# Prompt Gemini
prompt = f"""
You are a financial document question-answering assistant.

Answer the user's question using ONLY the provided document context.

Rules:
1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer cannot be found in the context, say:
   "I could not find this information in the document."
4. Give a clear and concise answer.
5. Mention the relevant page number when possible.

User question:
{query}

Document context:
{context}
"""

# Generate answer
response = gemini_client.models.generate_content(
    model="gemini-3.8-flash",
    contents=prompt
)

# Display answer
print("\n" + "=" * 70)
print("ANSWER")
print("=" * 70)

print(response.text)

# Display sources
print("\n" + "=" * 70)
print("SOURCES")
print("=" * 70)

for i, result in enumerate(results):
    print(
        f"Source {i + 1}: "
        f"Page {result.payload['page_number']} "
        f"| Similarity: {result.score:.3f}"
    )

qdrant_client.close()