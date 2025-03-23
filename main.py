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

# === Config ===
CHROMA_ZIP_URL = "https://www.dropbox.com/scl/fi/ylt1a2653pq856glbv6xd/ChromaDB.zip?rlkey=z5n4k2q5rklxi6df1fitnsqeo&st=kyi8gqjq&dl=1"
CHROMA_DIR = "./ChromaDB"
COLLECTION_NAME = "brightside_supplement_data"
MODEL_NAME = "intfloat/e5-small-v2"

# === Step 1: Download + Extract ChromaDB if Missing ===
def download_and_extract_chroma():
    if not os.path.exists(CHROMA_DIR):
        print("🔽 ChromaDB not found. Downloading...")
        try:
            zip_path = "ChromaDB.zip"

            # Streaming download (safe for large files)
            with requests.get(CHROMA_ZIP_URL, stream=True) as r:
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            # Extract to ./ChromaDB
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(".")
            print("✅ ChromaDB extracted successfully.")

        except Exception as e:
            print("❌ Failed to download or unzip ChromaDB:", str(e))
            raise

download_and_extract_chroma()

# === Step 2: Initialize FastAPI ===
app = FastAPI()

# === Step 3: Define Input Schema ===
class QueryRequest(BaseModel):
    query: str
    top_n: int = 5

# === Step 4: Load ChromaDB Collection ===
embedding_fn = SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)

# === Utility to Clean Text ===
def clean_text(text: str) -> str:
    cleaned = re.sub(r"\(cid:\d+\)", "", text)
    cleaned = re.sub(r"[\u0000-\u001F\u007F-\u009F]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()

# === Step 5: Product Search Logic ===
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

# === Step 6: Expose POST /query Endpoint ===
@app.post("/query")
def query_products(req: QueryRequest):
    try:
        results = query_top_products(req.query, req.top_n)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
