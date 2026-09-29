from sentence_transformers import CrossEncoder

# Load reranker model
reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)

# Test question
query = "What was Apple's total net sales in 2025?"

# Example document chunks
documents = [
    "Apple's total net sales in 2025 were $416,161 million.",
    "Apple designs and manufactures smartphones and computers.",
    "Apple's research and development expenses increased in 2025."
]

# Create question-document pairs
pairs = []

for document in documents:
    pairs.append([query, document])

# Calculate relevance scores
scores = reranker.predict(pairs)

# Display results
for i in range(len(documents)):

    print("=" * 70)

    print("Document:", i + 1)

    print("Reranker Score:", round(float(scores[i]), 4))

    print("Text:")
    print(documents[i])