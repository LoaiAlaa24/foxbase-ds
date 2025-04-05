import base64
import pickle
import os
from IPython.display import Image, display
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.core import clean


class DocumentProcessor:
    def __init__(self, model,  pdf_path, output_dir="./doc_data/images/"):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.chunks = []
        self.texts = []
        self.tables = []
        self.images = []
        self.model = model
        self._load_pdf()
        

    def _load_pdf(self):
        """Partition and clean the PDF content."""
        
        self.raw_chunks = self._create_chunks()
        
        self.chunks = self._clean_chunks(self.raw_chunks)
        self._extract_text_and_tables()
        self.images = self._extract_images()
        
    def _create_chunks(self):
        raw_chunks = partition_pdf(
            filename=self.pdf_path,
            extract_images_in_pdf=True,
            extract_image_block_to_payload=True,
            extract_image_block_types=["Image", "Table"],
            infer_table_structure=True,
            pdf_infer_table_structure= True,
            chunking_strategy="by_title",
            multipage_sections=True,
            strategy="hi_res",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=1000,
            languages=["deu"],
            image_output_dir_path=self.output_dir
        )
        
        return raw_chunks

    def _clean_chunks(self, chunks):
        """Cleans extracted PDF chunks using unstructured.cleaners.core."""
        cleaned_chunks = []
        for chunk in chunks:
            if hasattr(chunk, "text"):
                chunk.text = clean(chunk.text, bullets=True,extra_whitespace=True,\
                        trailing_punctuation=True)
                cleaned_chunks.append(chunk)
        return cleaned_chunks

    def _extract_text_and_tables(self):
        """Separate text chunks and tables from extracted PDF data."""
        for chunk in self.chunks:
            if "CompositeElement" in str(type(chunk)):
                self.texts.append(chunk)

            for subchunk in chunk.metadata.orig_elements:
                if "Table" in str(type(subchunk)):
                    self.tables.append(subchunk)


    def summarize_tables(self, model):
        """Generate a detailed summarization of tables, including structure and key insights."""
        
        prompt_text = """
        You are an assistant tasked with summarizing tables in German language.
        
        Give a concise summary of table.
        - Reply in German Language only, and in a paragraph form.
        - If numerical values exist, describe their ranges and variations.
        - Reply in less than 300 tokens.
        
        Respond only with the concise summary of the description. 
        Do not include introductory phrases like "Here is the analysis."
        Just give the summary as it is.
        
        Table: {table_html}
        """

        prompt = ChatPromptTemplate.from_template(prompt_text)
        analyze_chain = {"table_html": lambda x: x} | prompt | model | StrOutputParser()

        tables_html = [table.metadata.text_as_html for table in self.tables]
        table_descriptions = analyze_chain.batch(tables_html, {"max_concurrency": 1})

        return table_descriptions

    def summarize(self, model):
        """Summarize extracted text and tables in German using ChatGroq."""
        prompt_text = """
        You are an assistant tasked with summarizing text in German language.
        Give a concise summary of text.

        Respond only with the summary, no additional comment.
        Do not start your message by saying "Here is a summary" or anything like that.
        Just give the summary as it is.
        
        - Reply in less than 300 tokens.

        Table or text chunk: {element}
        """
        prompt = ChatPromptTemplate.from_template(prompt_text)
        summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

        # Summarize text chunks
        text_summaries = summarize_chain.batch(self.texts, {"max_concurrency": 1})
        
        return text_summaries
