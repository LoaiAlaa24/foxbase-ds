from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from base64 import b64decode
import logging

# Configure the logging system
logging.basicConfig(level=logging.INFO)  # Set the logging level

class ChatBotEngine:
    def __init__(self, retriever, model):
        """
        Initializes the chatbot engine with a retriever and an LLM model.

        :param retriever: A document retriever for fetching relevant context.
        :param model_name: The name of the OpenAI model to use.
        """
        self.retriever = retriever
        self.model = model
        self.chain = self._build_chain()

    def _retrieve(self, inputs):
        
        question = inputs["question"]

        return self.retriever.invoke(question)

    def _parse_docs(self, docs):
        """Split base64-encoded images and texts."""
        b64_images = []
        text_elements = []

        for doc in docs:
            try:
                b64decode(doc)
                b64_images.append(doc)
            except Exception:
                text_elements.append(doc)

        return {"images": b64_images, "texts": text_elements}

    def _build_prompt(self, kwargs):
        """Builds a prompt using the retrieved context and user question."""
        
        docs_by_type = kwargs["context"]
        user_question = kwargs["question"]

        context_text = ""
        if docs_by_type["texts"]:
            for text_element in docs_by_type["texts"]:
                context_text += text_element.page_content + "\n"

        prompt_template = f"""
        Answer the question based only on the following context, which can include text, and tables.
        
        Context: {context_text}
        Question: {user_question}
        
        Always reply in German
        Explain in paragraph the answer and the values the table provided in the context.
        Reference the context used to answer between double quotations.
        
        If the Question is not related to the content, reply saying that you can not answer this question.
        
        """

        prompt_content = [{"type": "text", "text": prompt_template}]

        return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])

    def _build_chain(self):
        """Constructs the chatbot processing chain."""
        
        chain = (
            {  
                "context": RunnableLambda(self._retrieve) 
                    | RunnableLambda(self._parse_docs),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(self._build_prompt)
            | ChatOpenAI(model="gpt-4o-mini")
            | StrOutputParser()
        )

        return chain

    async def astream_ask(self, question):
        """Streams the chatbot's response asynchronously."""
        
        async for chunk in self.chain.astream({"question": question}):
            yield chunk
