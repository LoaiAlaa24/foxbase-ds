import os
import uuid
import pickle
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema import Document
from langchain_community.storage import MongoDBStore
from utils.document_processor import DocumentProcessor

HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

PDF_PATH = os.getenv("PDF_FILE_PATH")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

MONGODB_URL = os.getenv("MONGODB_URL")
MONGODB_NAME = os.getenv("MONGODB_NAME")
MONGODB_COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME")

class RetrievalCreatorPipeline:
    def __init__(self, model, pdf_path: str= PDF_PATH):
        self.model = model 
        
        # Process document
        self.processor = DocumentProcessor(model=self.model, pdf_path=pdf_path)
        self.texts = self.processor.texts
        self.tables = self.processor.tables
        
        # Summarize extracted text and tables
        self.text_summaries = self.processor.summarize()
        self.table_summaries = self.processor.summarize_tables()
        
        self.table_ids = [str(uuid.uuid4()) for _ in self.table_summaries]
        self.doc_ids = [str(uuid.uuid4()) for _ in self.text_summaries]
        
        # create the open-source embedding function
        embedding_function = HuggingFaceInferenceAPIEmbeddings(
            api_key=HUGGING_FACE_API_KEY, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # Initialize vector storage and BM25 storage
        self.vectorstore = Chroma(collection_name=COLLECTION_NAME,\
            embedding_function=embedding_function, persist_directory=CHROMA_DB_PATH)
        
        self.store  = MongoDBStore(MONGODB_URL, db_name=MONGODB_NAME,
                             collection_name=MONGODB_COLLECTION_NAME)

        self.id_key = "doc_id"
        
        # Initialize retriever
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            id_key=self.id_key
        )
        
        
        # Add summarized text and tables to retriever
        self._add_text_summaries_to_retriever()
        self._add_table_summaries_to_retriever()
        self._initialize_bm25_retriever()
    
    def _add_text_summaries_to_retriever(self):

        # Convert summaries into LangChain Document objects
        summary_texts = [
            Document(page_content=summary, metadata={self.id_key: self.doc_ids[i]})
            for i, summary in enumerate(self.text_summaries)
        ]
        
        # Add to Chroma retriever
        self.retriever.vectorstore.add_documents(summary_texts)
        
        
        # Create LangChain Documents
        text_documents = [Document(page_content=text.text, metadata=text.metadata.to_dict()) for text in self.texts]
        self.retriever.docstore.mset(list(zip(self.doc_ids, text_documents)))
    
    def _add_table_summaries_to_retriever(self):
        # Generate unique IDs for table summaries
        
        # Convert summaries into LangChain Document objects
        summary_tables = [
            Document(page_content=summary, metadata={self.id_key: self.table_ids[i]})
            for i, summary in enumerate(self.table_summaries)
        ]
        # Add to Chroma retriever
        self.retriever.vectorstore.add_documents(summary_tables)
        
        table_documents = [Document(page_content=table.text, metadata=table.metadata.to_dict()) for table in self.tables]
        self.retriever.docstore.mset(list(zip(self.table_ids, table_documents)))
    
    def _initialize_bm25_retriever(self):
        # Prepare documents for BM25 retrieval
        
        self.bm25_documents=[
            Document(page_content=str(text.text), metadata={self.id_key: self.doc_ids[i]}) for i, text in enumerate(self.texts)
        ] + [
            Document(page_content=str(table.text), metadata={self.id_key: self.table_ids[j]}) for j, table in enumerate(self.tables)
        ]
        
        self.bm25_retriever = BM25Retriever.from_documents(self.bm25_documents)
            