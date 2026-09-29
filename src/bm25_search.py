from rank_bm25 import BM25Okapi

from ingestion import extract_text_from_pdf, create_chunks


# Load PDF
pdf_path = "data/documents/company_report.pdf"

pages = extract_text_from_pdf(pdf_path)

# Create chunks
chunks = create_chunks(pages)

print("Number of chunks:", len(chunks))


# Get text from each chunk
documents = []

for chunk in chunks:
    documents.append(chunk["text"])


# Tokenize documents
tokenized_documents = []

for document in documents:
    tokens = document.lower().split()
    tokenized_documents.append(tokens)


# Create BM25 index
bm25 = BM25Okapi(tokenized_documents)


# Ask question
query = input("\nEnter your question: ")

# Tokenize question
query_tokens = query.lower().split()


# Calculate BM25 scores
scores = bm25.get_scores(query_tokens)


# Get top 5 results
top_indices = scores.argsort()[-5:][::-1]


print("\n" + "=" * 70)
print("BM25 SEARCH RESULTS")
print("=" * 70)


for i, index in enumerate(top_indices):

    print("\n" + "-" * 70)

    print("Result:", i + 1)

    print("Page:", chunks[index]["page_number"])

    print("BM25 Score:", round(scores[index], 3))

    print("\nText:")

    print(chunks[index]["text"])