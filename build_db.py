import os
import chromadb
from sentence_transformers import SentenceTransformer


chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection("startup_knowledge")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text):
    return embed_model.encode(text).tolist()

def build_db():
    folder = "./knowledge_base"
    files = os.listdir(folder)
    
    added = 0
    for i, filename in enumerate(files):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            
            vector = embed_text(text)
            
            collection.add(
                documents=[text],
                embeddings=[vector],
                ids=[f"doc_{i}"],
                metadatas=[{"filename": filename}]
            )
            print(f"Added: {filename}")
            added += 1
    
    print(f"Database built with {added} documents")

build_db()