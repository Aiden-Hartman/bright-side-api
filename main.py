from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import chromadb
import re
import os
import zipfile
import requests
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from collections import defaultdict

# --- Download vector DB from external source if needed ---
CHROMA_ZIP_URL = "https://huggingface.co/datasets/aidenbrightside/files/resolve/main/ChromaDB.zip"
CHROMA_DIR = "./ChromaDB"

def download_and_extract_chroma():
    if not os.path.exists(CHROMA_DIR):
        print("🔽 ChromaDB not found. Downloading...")
        try:
            r = requests.get(CHROMA_ZIP_URL)
            zip_path = "ChromaDB.zip"
            with open(zip_path, "wb") as f:
                f.write(r.content)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(".")
            print("✅ ChromaDB extracted.")
        except Exception as e:
            print("❌ Failed to download ChromaDB:", str(e))
            raise

download_and_extract_chroma()

# --- Config ---
COLLECTION_NAME = "brightside_supplement_data"
MODEL_NAME = "intfloat/e5-small-v2"

# --- Initialize FastAPI ---
app = FastAPI()

# --- Request Schema ---
class QueryRequest(BaseModel):
    query: str
    top_n: int = 5

# --- Load Vector DB ---
embedding_fn = SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)

def clean_text(text: str) -> str:
    cleaned = re.sub(r"\(cid:\d+\)", "", text)
    cleaned = re.sub(r"[\u0000-\u001F\u007F-\u009F]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()

def query_top_products(query: str, top_n: int = 5) -> List[Dict[str, Any]]:
    search_results = collection.query(
        query_texts=[query],
        n_results=30
    )
    
    product_hits = {}
    grouped_chunks = defaultdict(list)

    for doc, metadata, distance in zip(
        search_results["documents"][0],
        search_results["metadatas"][0],
        search_results["distances"][0]
    ):
        filename = metadata.get("filename", "Unknown")
        product = metadata.get("product", filename)
        source_type = metadata.get("source_type", "unknown")

        if product not in product_hits or distance < product_hits[product]["match_score"]:
            product_hits[product] = {
                "product": product,
                "filename": filename,
                "source_type": source_type,
                "match_score": distance,
                "matched_chunk": clean_text(doc),
                "full_document": ""
            }

        grouped_chunks[product].append(doc)

    for product in product_hits:
        full_chunks = grouped_chunks[product]
        full_text = "\n\n".join(clean_text(chunk) for chunk in full_chunks)
        product_hits[product]["full_document"] = full_text

    sorted_hits = sorted(product_hits.values(), key=lambda x: x["match_score"])
    return sorted_hits[:top_n]

# --- Endpoint ---
@app.post("/query")
def query_products(req: QueryRequest):
    try:
        results = query_top_products(req.query, req.top_n)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
