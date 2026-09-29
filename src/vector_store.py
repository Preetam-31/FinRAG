import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


# --------------------------------
# 1. Extract PDF text
# --------------------------------

def extract_text_from_pdf(pdf_path):

    document = fitz.open(pdf_path)

    pages = []

    for page_number in range(len(document)):

        page = document[page_number]

        text = page.get_text()

        pages.append({
            "page_number": page_number + 1,
            "text": text
        })

    document.close()

    return pages


# --------------------------------
# 2. Create chunks
# --------------------------------

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
                "page_number": page["page_number"],
                "text": chunk
            })

    return chunks


# --------------------------------
# 3. Load embedding model
# --------------------------------

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------
# 4. Load PDF and create chunks
# --------------------------------

pdf_path = "data/documents/company_report.pdf"

pages = extract_text_from_pdf(pdf_path)

chunks = create_chunks(pages)

print("Number of pages:", len(pages))

print("Number of chunks:", len(chunks))


# --------------------------------
# 5. Create embeddings
# --------------------------------

texts = []

for chunk in chunks:
    texts.append(chunk["text"])


embeddings = model.encode(texts)

print("Embedding shape:", embeddings.shape)


# --------------------------------
# 6. Create Qdrant client
# --------------------------------

client = QdrantClient(
    path="qdrant_data"
)


# --------------------------------
# 7. Create collection
# --------------------------------

collection_name = "finrag_documents"


if not client.collection_exists(collection_name):

    client.create_collection(
        collection_name=collection_name,

        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )


# --------------------------------
# 8. Create Qdrant points
# --------------------------------

points = []

for i in range(len(chunks)):

    points.append(
        PointStruct(
            id=i,
            vector=embeddings[i].tolist(),

            payload={
                "text": chunks[i]["text"],
                "page_number": chunks[i]["page_number"],
                "document": "company_report.pdf"
            }
        )
    )


# --------------------------------
# 9. Store points in Qdrant
# --------------------------------

client.upsert(
    collection_name=collection_name,
    points=points
)


print("Successfully stored chunks in Qdrant!")

print("Total vectors stored:", len(points))