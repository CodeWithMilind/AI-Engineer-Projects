from langchain_huggingface import HuggingFaceEmbeddings

# Load embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Chunks from your PDF
chunks = [
    "Python is a programming language.",
    "AI is transforming healthcare.",
    "Cats are domestic animals."
]

# Generate embeddings
vectors = embedding.embed_documents(chunks)

print(len(vectors))        # 3 vectors
print(len(vectors[0]))     # 384 dimensions
print(vectors[0][:5])      # First 5 numbers
print("hello")