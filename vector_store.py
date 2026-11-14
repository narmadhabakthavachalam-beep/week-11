import os
import csv
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Document folder
document_folder = "dataset/documents"

documents = []
ids = []
metadatas = []

# Read all text documents
for filename in os.listdir(document_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(document_folder, filename)

        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        documents.append(content)
        ids.append(filename.replace(".txt", ""))

        title = content.split("\n")[0].strip()

        metadatas.append({
            "title": title,
            "category": "Technology",
            "source": filename
        })

# Generate embeddings
embeddings = model.encode(documents)

# Save embeddings
np.save("dataset/embeddings.npy", embeddings)

# Create ChromaDB client
client = chromadb.PersistentClient(path="chroma_db")

# Create collection
collection = client.get_or_create_collection(
    name="technology_documents"
)

# Store documents and embeddings
collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)

print("Documents and embeddings stored successfully!")
print("Total documents:", len(documents))