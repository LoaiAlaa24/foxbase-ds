import os

from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

class EmbeddingModelInit:
    
    def __init__(self):
                    
        self.embedding_model = HuggingFaceInferenceAPIEmbeddings(
            api_key=HUGGING_FACE_API_KEY, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
