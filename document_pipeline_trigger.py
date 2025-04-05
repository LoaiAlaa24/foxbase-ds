from utils.llm_initializer import LLMInit
from retrieval_core.retrieval_creator_pipeline import RetrievalCreatorPipeline

llm_initializer = LLMInit()

model = llm_initializer.llm

pipeline = RetrievalCreatorPipeline(model = model)
