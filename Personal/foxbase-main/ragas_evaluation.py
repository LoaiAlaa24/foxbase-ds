import asyncio
import logging
import pandas as pd

from langfuse import Langfuse
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    SemanticSimilarity,
    AnswerAccuracy
)
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings

from utils.llm_initializer import LLMInit
from utils.emb_model_initializer import EmbeddingModelInit
from retrieval_core.retrieval_inference import RetrievalInference
from retrieval_core.chatbot_engine import ChatBotEngine

# Initialize components
llm_initializer = LLMInit()
emb_model = EmbeddingModelInit()
retrieval_inference = RetrievalInference(embedding_model=emb_model.embedding_model)
chatbot_engine = ChatBotEngine(retriever=retrieval_inference.ensemble_retriever, model=llm_initializer.llm)

# Configure the logging system
logging.basicConfig(level=logging.INFO)  # Set the logging level

langfuse = Langfuse()

langfuse.auth_check()

# metrics you chose
metrics = [
    Faithfulness(),
    ResponseRelevancy(),
    LLMContextPrecisionWithoutReference(),
    SemanticSimilarity(),
    AnswerAccuracy()
]


# Load evaluation dataset
df = pd.read_csv("evaluation_dataset/questions_answers.csv")

# Initialize similarity model
model = ChatOpenAI(model="gpt-4o-mini")

# util function to init Ragas Metrics
def init_ragas_metrics(metrics, llm, embedding):
    for metric in metrics:
        if isinstance(metric, MetricWithLLM):
            metric.llm = llm
        if isinstance(metric, MetricWithEmbeddings):
            metric.embeddings = embedding
        run_config = RunConfig()
        metric.init(run_config)
    
async def score_with_ragas(query, chunks, answer, expected_answer):
    scores = {}
    for m in metrics:
        sample = SingleTurnSample(
            user_input=query,
            retrieved_contexts=chunks,
            response=answer,
            reference=expected_answer
        )
        logging.info(f"calculating {m.name}")
        scores[m.name] = await m.single_turn_ascore(sample)
    return scores

async def evaluate():
    
    # start a new trace when you get a question
    for index, row in df.iterrows():
        
        query = row["Frage"]
        expected_answer = row["Antwort"]
    
        trace = langfuse.trace(name = "new_rag")

        # retrieve the relevant chunks
        chunks = retrieval_inference.ensemble_retriever.invoke(query)
        context = [chunk.page_content for chunk in chunks]
        
        # # pass it as span
        trace.span(
            name = "retrieval", input={'question': query}, output={'contexts': context}
        )
        
        # Get chatbot response
        answer = ""
        async for chunk in chatbot_engine.astream_ask(question=query):
            answer += chunk  # Concatenate streamed response chunks
        
        trace.span(
            name = "generation", input={'question': query, 'contexts': context}, output={'answer': answer}
        )

        # compute scores for the question, context, answer tuple
        scores = await score_with_ragas(query, context, answer, expected_answer)
        
        # Log scores using .score()
        for metric_name, value in scores.items():
            trace.score(name=f"ragas_{metric_name}", value=value)

init_ragas_metrics(
    metrics,
    llm=LangchainLLMWrapper(model),
    embedding=LangchainEmbeddingsWrapper(emb_model.embedding_model),
)

# Run the evaluation
asyncio.run(evaluate())
