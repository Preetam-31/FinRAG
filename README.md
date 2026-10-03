# FinRAG — Financial Retrieval-Augmented Generation System

FinRAG is a Retrieval-Augmented Generation (RAG) system designed to answer questions from financial reports.

The project uses Apple's 2025 Form 10-K as the source document and combines keyword-based retrieval, dense vector retrieval, Reciprocal Rank Fusion (RRF), and CrossEncoder reranking to retrieve relevant information before generating an answer.

---

## Project Overview

Financial reports contain a large amount of structured and unstructured information, making it difficult to quickly find specific financial data.

FinRAG addresses this problem by:

1. Extracting text from financial documents.
2. Splitting the document into smaller chunks.
3. Retrieving relevant chunks using both keyword and semantic search.
4. Combining retrieval results using Reciprocal Rank Fusion.
5. Reranking retrieved chunks using a CrossEncoder.
6. Providing the most relevant context to an LLM.
7. Generating an answer based on the retrieved information.

---

## Architecture

```text
                Financial Report
                      │
                      ▼
               PDF Text Extraction
                      │
                      ▼
                 Text Chunking
                      │
              ┌───────┴───────┐
              ▼               ▼
           BM25            Embeddings
        Retrieval          Retrieval
              │               │
              │               ▼
              │            Qdrant
              │         Vector Search
              │               │
              └───────┬───────┘
                      ▼
             Reciprocal Rank Fusion
                      │
                      ▼
             CrossEncoder Reranking
                      │
                      ▼
                Top Retrieved
                   Context
                      │
                      ▼
                  LLM
                      │
                      ▼
                 Final Answer

---
## Key Features

- Financial document question answering
- PDF document ingestion
- Document chunking
- BM25 keyword retrieval
- Dense semantic retrieval
- Qdrant vector database
- Reciprocal Rank Fusion (RRF)
- CrossEncoder reranking
- LLM-based answer generation
- Retrieval evaluation using Recall@5 and MRR
- RAGAS-based faithfulness evaluation

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| LLM | Gemini / Ollama |
| Keyword Retrieval | BM25 |
| Embeddings | Sentence Transformers |
| Vector Database | Qdrant |
| Reranker | BAAI/bge-reranker-base |
| Evaluation | RAGAS |
| Document | Apple's 2025 Form 10-K |
| Frontend | React + Vite |

---

## Retrieval Pipeline

### 1. Document Ingestion

The financial report is loaded and its text is extracted from the PDF.

### 2. Chunking

The extracted document is divided into smaller chunks so that relevant sections can be retrieved efficiently.

The current document contains approximately **426 chunks**.

### 3. BM25 Retrieval

BM25 is used for keyword-based retrieval.

It is useful when a question contains exact financial terminology, numbers, or keywords appearing in the document.

### 4. Dense Retrieval

The question is converted into an embedding using a Sentence Transformer model.

The embedding is compared against document vectors stored in Qdrant to retrieve semantically similar chunks.

### 5. Reciprocal Rank Fusion

BM25 and dense retrieval produce separate rankings.

RRF combines these rankings to produce a hybrid retrieval ranking.

### 6. CrossEncoder Reranking

The top candidate chunks are passed through a CrossEncoder reranker.

The reranker evaluates the relevance of each question-context pair and produces a refined ranking.

### 7. LLM Answer Generation

The highest-ranked contexts are provided to the language model, which generates the final answer based on the retrieved financial information.

---

## Evaluation

The retrieval pipeline was evaluated using **20 financial question-answering questions** based on Apple's 2025 Form 10-K.

### Retrieval Results

| Metric | Result |
|---|---:|
| Recall@5 | **90%** |
| MRR | **0.72** |

### Faithfulness

A previous RAGAS evaluation recorded:

**Faithfulness: 85%**

This value is currently considered **provisional** and will be re-evaluated in a future evaluation run.

### Interpretation

- **Recall@5 = 90%** means the relevant information was retrieved within the top five results for 18 of the 20 evaluation questions.
- **MRR = 0.72** measures how highly the first relevant result was ranked.
- **Faithfulness = 85%** indicates the previous RAGAS evaluation found a high degree of alignment between generated answers and retrieved context.

---

## Project Structure

```text
FinRAG/
│
├── data/
│   └── documents/
│       └── company_report.pdf
│
├── evaluation/
│   ├── questions.json
│   ├── evaluate_rag.py
│   ├── retrieval_metrics.py
│   └── retrieval_metrics.csv
│
├── frontend/
│
├── src/
│   ├── ingestion.py
│   └── ...
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md                 

