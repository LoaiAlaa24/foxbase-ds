import os
import logging

from pymongo import MongoClient
from langchain.vectorstores import Chroma
from langchain_community.storage import MongoDBStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers import BM25Retriever
from langchain.schema.document import Document

logging.basicConfig(level=logging.INFO)  # Set the logging level

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")
BM25_SAVE_PATH = os.getenv("BM25_SAVE_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

MONGODB_URL = os.getenv("MONGODB_URL")
MONGODB_NAME = os.getenv("MONGODB_NAME")
MONGODB_COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME")
MONGODB_LOCAL_PATH = os.getenv("MONGODB_LOCAL_PATH")

client = MongoClient(MONGODB_URL)  # Update if using a different host
db = client[MONGODB_NAME]  # Replace with your database name
collection_name = MONGODB_COLLECTION_NAME

class RetrievalInference:
    def __init__(self, embedding_model):
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME, 
            embedding_function=embedding_model, 
            persist_directory=CHROMA_DB_PATH
        )
        
        self.docstore = MongoDBStore(
            MONGODB_URL, 
            db_name=MONGODB_NAME,
            collection_name=MONGODB_COLLECTION_NAME
        )
        
        self.id_key = "doc_id"
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            id_key=self.id_key
        )
        
        self.populate_bm25()

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.retriever],
            weights=[0.5, 0.5]
        )

         
    def populate_bm25(self):   
    
        loaded_list = []
        
        with open("./doc_data/bm25_text.txt", "r") as file:
            loaded_list = file.read().splitlines()
            
        bm25_docs = [Document(page_content= text) for text in loaded_list]

        # Create BM25Retriever
        self.bm25_retriever = BM25Retriever.from_documents(bm25_docs, k=2)
