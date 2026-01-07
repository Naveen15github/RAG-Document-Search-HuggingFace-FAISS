# RAG-Document-Search-HuggingFace-FAISS

## Overview

![Alt text](image_url_or_path)

In this Mini project, I built a **Retrieval-Augmented Generation (RAG)** system that allows me to **search my documents locally** using embeddings, without relying on any paid APIs. I implemented it fully myself, using a text file as the source, splitting it into chunks, embedding the chunks with a HuggingFace model, and searching for relevant content with **FAISS vector search**.

This project helped me understand **RAG, embeddings, and vector-based search**, and everything works fully offline.

---

## Features

* **Local Embeddings**: I used HuggingFace’s `all-MiniLM-L6-v2` model to generate embeddings for my document chunks.
* **Vector Search**: I implemented FAISS to index the chunks and quickly find the most relevant content.
* **Document Chunking**: I split the documents into smaller chunks (300 characters each, 50-character overlap) to improve search accuracy.
* **Query Handling**: I added a check to print “Information not available in the documents” if the query isn’t found.
* **Offline & Free**: Everything runs locally, so I don’t need an API key or subscription.

---

## Project Structure

```
rag-project/
│
├─ docs/
│   └─ notes.txt           # My source text document for testing
│
├─ rag.py                  # My main Python script implementing RAG
│
├─ requirements.txt        # Python dependencies
│
└─ README.md               # Project documentation
```
## How I Implemented It

1. **Load Document**: I read the content of `notes.txt` using Python’s `open()` function.
2. **Chunking**: I split the document into chunks of 300 characters with 50-character overlap using `CharacterTextSplitter` from LangChain.
3. **Embedding**: I generated embeddings for each chunk using HuggingFace’s `all-MiniLM-L6-v2`.
4. **Vector Store**: I built a FAISS vector store to store embeddings for fast similarity search.
5. **Query Search**: When I input a query, it is transformed into an embedding and matched against the vector store.

---

## Installation

1. **Create a virtual environment**:

```bash
python -m venv venv
```

2. **Activate the virtual environment**:

* Windows:

```bash
venv\Scripts\activate
```

* Mac/Linux:

```bash
source venv/bin/activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

> I used `langchain`, `faiss-cpu`, and `sentence-transformers` for this project.

---

## Usage

1. **Add your text documents** in the `docs` folder (e.g., `notes.txt`).

2. **Edit the query** in `rag.py` to search something you want:

```python
query = "what is AI"
```

3. **Run the script**:

```bash
python rag.py
```

```
Answer:

Artificial Intelligence (AI) is a branch of computer science
that focuses on creating systems capable of performing tasks
that normally require human intelligence.
```
![Alt text](image_url_or_path)
![Alt text](image_url_or_path)

---

## Example

**Document (`notes.txt`)**:

```
Artificial Intelligence (AI) is a branch of computer science...
Machine Learning is a subset of AI...
Deep Learning is a subset of Machine Learning...
```

**Query**:

```python
query = "what is AI"
```

**Output**:

```
Artificial Intelligence (AI) is a branch of computer science
that focuses on creating systems capable of performing tasks
that normally require human intelligence.
```

---

## What I Learned

By doing this project, I gained hands-on experience with:

* Using **HuggingFace embeddings** for text representation.
* Splitting documents into smaller chunks for better retrieval.
* Indexing and searching with **FAISS**.
* Building a basic **RAG pipeline** locally.


---

## Future Plans

* Support **multiple document types** like PDFs, Word files, or CSVs.
* Build a **chat interface** on top of this RAG system for conversational search.
* Return **top-k results** instead of just one.
* Add **caching** to avoid recalculating embeddings for unchanged documents.

---

## References

* [HuggingFace Transformers](https://huggingface.co/transformers/)
* [Sentence Transformers](https://www.sbert.net/)
* [FAISS Documentation](https://faiss.ai/)
* [LangChain Documentation](https://www.langchain.com/)

---

## Author

**Naveen G** – Implemented this project to learn RAG, embeddings, and local vector search.

---

If you want, I can also **write a ready-to-use `requirements.txt`** for this exact setup and add **LinkedIn-style badges** to make your GitHub repo look more professional.

Do you want me to do that?
