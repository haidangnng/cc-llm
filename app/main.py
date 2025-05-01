from fastapi import Body
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import requests
import chromadb
import os
import tempfile
from typing import List
from contextlib import asynccontextmanager
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

from .database.core import create_vector_store
from .entities.test import GenerateQuestionRequest, GenerateAnswerRequest

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
                raise HTTPException(
                    status_code=500, detail=f"Failed to load PDF: {e}")

            for d in docs:
                d.metadata["material_id"] = str(file.id)
                d.metadata["source_url"] = file.url

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)

            os.remove(tmp_path)

        try:
            vector_store.add_documents(all_chunks)
        except Exception as e:
            print(f"Error: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to save to vector store: {e}"
            )

        print("===== SUCCESS ====")

        return {"status": "success", "chunks_added": len(all_chunks)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-question")
def generate_questions(data: GenerateQuestionRequest):
    print("===== GENERATE_QUESTIONS ====")

    try:
        if not vector_store:
            raise HTTPException(
                status_code=500, detail="Vector store not initialized.")

        # Configure retriever with search parameters
        retriever = vector_store.as_retriever(search_kwargs={"k": 20})

        # Apply filter if materialIds are provided
        if data.materialIds:
            filter_query = {"material_id": {
                "$in": [str(i) for i in data.materialIds]}}
            retriever.search_kwargs["filter"] = filter_query

        # Use a meaningful query to retrieve relevant documents
        query = f"Generate {data.count} exam questions from materials with IDs: {data.materialIds}"
        docs = retriever.invoke(query)

        print(
            f"Retrieved {len(docs)} documents for material IDs: {data.materialIds}")
        if not docs:
            raise HTTPException(
                status_code=404, detail="No relevant materials found.")

        # Construct prompt
        prompt = (
            f"Generate {data.count} multiple choice exam questions ONLY from materials with IDs: {data.materialIds or 'ALL'}.\n"
            "For each question, provide exactly four answer choices and indicate the correct one.\n"
            "Return the result as a JSON list, strictly in this format:\n\n"
            '[{"question": "...", "choices": ["A. ...", "B. ...", "C. ...", "D. ..."], "correctAnswer": "A. ..."}]\n\n'
        )
        context = "\n\n".join([doc.page_content for doc in docs])
        messages = [{"role": "user", "content": prompt + context}]

        chat = ChatOpenAI(model="gpt-4")
        response = chat.invoke(messages)
        print(response)

        return {"query": prompt, "result": response.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-answer")
def generate_answer(data: GenerateAnswerRequest):
    print(data)
    try:
        if not vector_store:
            raise HTTPException(
                status_code=500, detail="Vector store not initialized.")

        # Create retriever and apply material ID filter if provided
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        if data.materialIds:
            retriever.search_kwargs["filter"] = {
                "material_id": {"$in": [str(i) for i in data.materialIds]}
            }

        docs = retriever.invoke(data.question)

        if not docs:
            raise HTTPException(
                status_code=404, detail="No relevant materials found.")

        # Construct context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])

        # Format the prompt for multiple choice generation
        prompt = (
            f"Based only on the following context, answer the question with multiple choice options.\n"
            f"For the question below, provide exactly four answer choices and clearly indicate the correct one.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {data.question}\n\n"
            f"Return the result strictly as a JSON object in this format:\n"
            f'{{"Question": "...", "Choices": ["A. ...", "B. ...", "C. ...", "D. ..."], "CorrectAnswer": "A. ..."}}\n\n'
        )

        messages = [{"role": "user", "content": prompt}]
        chat = ChatOpenAI(model="gpt-4")
        response = chat.invoke(messages)

        # result = {"question": data.question, "result": response.content}

        # Return parsed JSON string or raw content for now
        return response.content

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
