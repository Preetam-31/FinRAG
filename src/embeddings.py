from sentence_transformers import SentenceTransformer


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


texts = [
    "Apple reported strong revenue growth in 2025.",
    "The company increased its operating expenses."
]


embeddings = model.encode(texts)


print("Number of texts:", len(texts))
print("Embedding shape:", embeddings.shape)
print("First embedding:")
print(embeddings[0])