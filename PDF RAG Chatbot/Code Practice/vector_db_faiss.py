from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# PDF chunks
chunks = [
    "Python is a programming language.",
    "Artificial Intelligence is changing healthcare.",
    "Cats are domestic animals."
]

# Create vector database
db = FAISS.from_texts(
    texts=chunks,
    embedding=embedding
)