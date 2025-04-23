from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def get_db(documents):
    return Chroma.from_documents(documents, OpenAIEmbeddings())


def similarity_search(documents, query):
    db = get_db(documents)
    docs = db.similarity_search(query)
    return docs


def similarity_search_by_vector(documents, query):
    db = get_db(documents)
    embedding_vector = OpenAIEmbeddings().embed_query(query)
    docs = db.similarity_search_by_vector(embedding_vector)
    return docs
