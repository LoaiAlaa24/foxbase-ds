from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

class LLMInit:
    
    def __init__(self):
        self.llm = ChatGroq(temperature=0.0, model="llama-3.1-8b-instant")
