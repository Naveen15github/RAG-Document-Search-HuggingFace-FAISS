from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# Load document
with open("docs/notes.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = text_splitter.split_text(text)

# Local embedding
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Build vector store
vector_store = FAISS.from_texts(chunks, embeddings)

# Query example
query = "what is Cloud Computing"
docs = vector_store.similarity_search(query)

print("Answer:\n")
print(docs[0].page_content)                 