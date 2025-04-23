import requests
import chromadb
import os
import tempfile
from typing import List
from contextlib import asynccontextmanager
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from typing import Union
from fastapi import FastAPI, HTTPException

from .database.core import create_vector_store

vector_store = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store
    vector_store = create_vector_store()
    yield


app = FastAPI(lifespan=lifespan)


class FileMetadata(BaseModel):
    url: str
    id: int


@app.post("/files/process")
def ingest_files(files: List[FileMetadata]):
    print(files)
    global vector_store
    all_chunks = []

    try:
        for file in files:
            # Download file to temp file
            response = requests.get(file.url)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=400, detail=f"Failed to fetch file from {file.url}"
                )

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name

            # Load and split PDF
            try:
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load PDF: {e}")

            for d in docs:
                d.metadata["material_id"] = file.id
                d.metadata["source_url"] = file.url

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
            print("====CHUNKS====")

            os.remove(tmp_path)

        try:
            print("===== ADDING DOCUMENTS ====")
            vector_store.add_documents(all_chunks)
            print("===== DOCUMENTS ADDED ====")
        except Exception as e:
            print("===== EXCEPTION CAUGHT ====")
            print(f"Error: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to save to vector store: {e}"
            )

        print("===== SUCCESS ====")

        return {"status": "success", "chunks_added": len(all_chunks)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    client = chromadb.HttpClient(host="localhost", port=8000)
    collection = client.get_or_create_collection(name="course_materials")

    results = collection.get(include=["documents", "metadatas"])

    print("Documents stored in ChromaDB:")
    for doc, meta in zip(results["documents"], results["metadatas"]):
        print("Doc Preview:", doc[:100])
        print("Metadata:", meta)
        print("-" * 40)

    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.get("/materials")
def read_embedded_materials(item: str):
    return
