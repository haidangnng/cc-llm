import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

embedding_function = OpenAIEmbeddings(
    model="text-embedding-3-large", openai_api_key=openai_api_key
)

chroma_client = chromadb.HttpClient(host="localhost", port=8000)
chroma_client.heartbeat()
collection = chroma_client.get_or_create_collection(name="course_materials")


def create_vector_store():
    vector_store = Chroma(
        client=chroma_client,
        collection_name="course_materials",
        embedding_function=embedding_function,
    )
    return vector_store
