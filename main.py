import logging

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from utils.llm_initializer import LLMInit
from utils.emb_model_initializer import EmbeddingModelInit

from retrieval_core.retrieval_inference import RetrievalInference
from retrieval_core.chatbot_engine import ChatBotEngine

from models.user_query import UserQuery

# Configure the logging system
logging.basicConfig(level=logging.INFO)  # Set the logging level

app = FastAPI()

llm_initializer = LLMInit()
emb_model = EmbeddingModelInit()
retrieval_inference = RetrievalInference(embedding_model=emb_model.embedding_model)
chatbot_engine = ChatBotEngine(retriever= retrieval_inference.ensemble_retriever, model=llm_initializer.llm )

@app.post("/chat")
async def chat_endpoint(user_input: UserQuery):
    """Handle user messages via API with streaming."""
    
    async def response_stream():
        async for chunk in chatbot_engine.astream_ask(question=user_input.query):
            yield chunk

    return StreamingResponse(response_stream(), media_type="text/plain")