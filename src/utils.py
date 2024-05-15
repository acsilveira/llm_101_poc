# import textwrap
# from IPython.display import Markdown
# import requests
# from bs4 import BeautifulSoup
# from PyPDF2 import PdfReader
# import pdfplumber
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
import parameters as general_parameters
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import PromptTemplate
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import HTMLHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
import sys
from langchain.llms import OpenAI
from logging_setter import logger


class UtilsLLM:
    """
    Set of useful transformations to handle LLM interactions.
    """

    def __init__(self):
        self.logger = logger
        self.logger.info("Utils initialized")

    # def to_markdown(self, text):
    #     """ Interpret string as markdown """
    #
    #     text = text.replace("•", "  *")
    #     return Markdown(textwrap.indent(text, "> ", predicate=lambda _: True))

    def get_text_from_web_article_parsing_html_langchain(self, url):
        """ Get text from a web URL article, parsing HTML with LangChain"""

        try:
            print(f"===>{sys.version}")
            response = requests.get(url)
            if response.status_code == 200:
                headers_to_split_on = [
                    ("h1", "Header 1"),
                    ("h2", "Header 2"),
                    ("h3", "Header 3"),
                    ("h4", "Header 4"),
                ]
                html_splitter = HTMLHeaderTextSplitter(
                    headers_to_split_on=headers_to_split_on
                )
                html_header_splits = html_splitter.split_text_from_url(url)
                log_msg = (
                    f"Succeed getting text from URL {url} and splitting in HTML headers"
                )
                return html_header_splits, log_msg
            else:
                log_msg = f"Failed to fetch content from URL {url}"
                self.logger.info(log_msg)
                return None, log_msg
        except Exception as e:
            log_msg = f"An error occurred: {e}"
            self.logger.error(log_msg)
            raise e

    def get_text_from_web_article_parsing_html(self, url):
        """ Get text from a web URL article, parsing HTML"""

        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                tag_elements = soup.find_all("p")
                text_extracted = ""
                for tag_i in tag_elements:
                    text_extracted += tag_i.get_text()
                log_msg = f"Succeed getting text from URL {url}"
                self.logger.info(log_msg)
                return text_extracted, log_msg
            else:
                log_msg = f"Failed to fetch content from URL {url}"
                self.logger.error(log_msg)
                return None, log_msg
        except Exception as e:
            log_msg = f"An error occurred: {e}"
            self.logger.error(log_msg)
            raise e

    def get_text_from_web_article(self, url):
        """ Get text from a web URL article """

        try:
            response = requests.get(url)
            # Check if request was successful
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                text = soup.get_text()
                # parsed_text = text.replace("\n", " ")
                log_msg = f"Succeed getting text from URL {url}"
                self.logger.info(log_msg)
                return text, log_msg
            else:
                log_msg = f"Failed to fetch content from URL {url}"
                self.logger.error(log_msg)
                return None, log_msg
        except Exception as e:
            log_msg = f"An error occurred: {e}"
            self.logger.error(log_msg)
            raise e

    # def read_pdf(self, file_path):
    #     """ Read a pdf file and get its content """
    #
    #     pdf_reader = PdfReader(file_path)
    #     file_text_content = pdf_reader
    #     text_content = ""
    #     number_of_pages = len(pdf_reader.pages)
    #     for i_page in range(number_of_pages):
    #         page = pdf_reader.pages[i_page]
    #         text_content += page.extract_text()
    #     return text_content

    # def read_pdf_pdfplumber(self, file_path):
    #     """ Read a pdf file and get its content using pdf plumber"""
    #
    #     with pdfplumber.open(file_path) as f:
    #         number_of_pages = len(f.pages)
    #         text_content = ""
    #         for i_page in range(number_of_pages):
    #             page = f.pages[i_page]
    #             text_content += page.extract_text()
    #     return text_content

    def define_embedding_model(self):
        try:
            embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        except Exception as e:
            raise e

        log_msg = "Succeed defining the embedding model"
        self.logger.info(log_msg)
        return embedding_model, log_msg

    def check_if_pinecone_index_exists(
            self, pinecone_client, par__vector_store_index_name
    ):
        if par__vector_store_index_name in pinecone_client.list_indexes().names():
            log_msg = "Pinecone index exists."
            self.logger.info(log_msg)
            return True, log_msg
        log_msg = "Pinecone index not found."
        self.logger.error(log_msg)
        return False, log_msg

    def create_pinecone_index(self, pinecone_client, par__vector_store_index_name):
        # If index already exists
        if self.check_if_pinecone_index_exists(
                pinecone_client, par__vector_store_index_name
        ):
            # Delete index before create it again
            try:
                pinecone_client.delete_index(par__vector_store_index_name)
            except Exception as e:
                log_msg = "Failed trying to delete pinecone index"
                self.logger.error(log_msg)
                raise e

        # Create index
        try:
            pinecone_client.create_index(
                name=par__vector_store_index_name,
                dimension=768,  # Replace with your model dimensions
                metric="euclidean",  # Replace with your model metric
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        except Exception as e:
            log_msg = "Failed creating pinecone index"
            self.logger.error(log_msg)
            raise e
        log_msg = "Succeed creating pinecone index"
        self.logger.info(log_msg)
        return True, log_msg

    def upload_vectors_to_vectorstore(
            self, par__vector_store_index_name, chunks, embedding_model
    ):
        # Upload vectors to Pinecone
        try:
            vectorstore_from_docs = PineconeVectorStore.from_documents(
                chunks,
                index_name=par__vector_store_index_name,
                embedding=embedding_model,
            )
        except Exception as e:
            log_msg = "Failed uploading vectors to vectorstore"
            self.logger.error(log_msg)
            raise e
        log_msg = "Succeed uploading vectors to vectorstore"
        self.logger.info(log_msg)
        return vectorstore_from_docs, log_msg

    def define_llm_model(self, llm_model_choice):
        if llm_model_choice == "Gemini":
            try:
                llm_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
                self.logger.info("LLM model defined as Gemini")
            except Exception as e:
                log_msg = "Failed trying the define the llm model."
                self.logger.error(log_msg)
                raise e
            log_msg = "Succeed defining the llm model"
            self.logger.info(log_msg)
            return llm_model, log_msg
        elif llm_model_choice == "chatGPT":
            try:
                llm_model = OpenAI(model_name="gpt-3.5-turbo-0125")
                self.logger.info("LLM model defined as chatGPT, gpt-3.5-turbo-0125")
            except Exception as e:
                log_msg = "Failed trying the define the llm model."
                self.logger.error(log_msg)
                raise e
            log_msg = "Succeed defining the llm model"
            self.logger.info(log_msg)
            return llm_model, log_msg
        else:
            log_msg = "Unknown LLM model choice."
            self.logger.error(log_msg)
            return None, log_msg

    def prepare_prompt(self):
        try:
            prompt = PromptTemplate(
                template=general_parameters.par__prompt_template,
                input_variables=[
                    general_parameters.par__prompt_template_var_context,
                    general_parameters.par__prompt_template_var_input,
                ],
            )
        except Exception as e:
            log_msg = "Failed preparing prompt"
            self.logger.error(log_msg)
            raise e
        log_msg = "Succeed preparing prompt"
        self.logger.info(log_msg)
        return prompt, log_msg

    def build_chain_gemini(self, vectorstore_from_docs, llm_model, prompt):
        try:
            retriever = vectorstore_from_docs.as_retriever()
            combine_docs_chain = create_stuff_documents_chain(llm_model, prompt)
            retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        except Exception as e:
            raise e
        log_msg = "Succeed building chain"
        self.logger.info(log_msg)
        return retrieval_chain, log_msg

    def build_chain_chat_gpt(self, vectorstore_from_docs, llm_model, prompt):
        try:
            # retrieval_chain = create_retrieval_chain(llm_model, vector_store_client, embedding_model)
            retriever = vectorstore_from_docs.as_retriever()
            combine_docs_chain = create_stuff_documents_chain(llm_model, prompt)
            retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        except Exception as e:
            raise e
        log_msg = "Succeed building chain"
        self.logger.info(log_msg)
        return retrieval_chain, log_msg

    def build_chain_wo_retriver(
            self, llm_model, prompt, par__prompt_template_var_input, input_question
    ):
        print(par__prompt_template_var_context + "=" + input_context[:500])
        try:
            result = llm_model.invoke(
                prompt.format(par__prompt_template_var_input=input_question)
            )
        except Exception as e:
            raise e
        log_msg = "Succeed asking question about content, without a retriever"
        self.logger.info(log_msg)
        return result, log_msg

    def asking_question_about_content_gemini(self, retrieval_chain, question):
        try:
            answer_about_content = retrieval_chain.invoke(
                {general_parameters.par__prompt_template_var_input: question}
            )
        except Exception as e:
            log_msg = "Failed asking question about content"
            self.logger.error(log_msg)
            raise e
        log_msg = "Succeed asking question about content for Gemini"
        self.logger.info(log_msg)
        return answer_about_content, log_msg

    def asking_question_about_content_chat_gpt(self, retrieval_chain, question):
        try:
            answer_about_content = retrieval_chain.invoke(
                {general_parameters.par__prompt_template_var_input: question}
            )
        except Exception as e:
            log_msg = "Failed asking question about content"
            self.logger.error(log_msg)
            raise e
        log_msg = "Succeed asking question about content for chatGPT"
        self.logger.info(log_msg)
        return answer_about_content, log_msg

    def wait_for(self, seconds_to_wait):
        """ Wait some seconds """

        self.logger.info(f"Waiting {seconds_to_wait} seconds...")
        time.sleep(seconds_to_wait)
        self.logger.info("...continuing now.")
        pass

    def split_text_into_chunks(self, raw_text_content):
        """ Split raw text into chunks """

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=general_parameters.par__chunk_size,
            chunk_overlap=general_parameters.par__chunk_overlap,
        )
        chunks = text_splitter.create_documents([raw_text_content])
        log_msg = "Text splitted into chunks with success."
        self.logger.info(log_msg)
        return chunks, log_msg

    def get_text_from_web_article_parsing_text(self, url):
        text_content, log_msg = self.get_text_from_web_article(url)
        self.logger.info(log_msg)
        chunks, log_msg = self.split_text_into_chunks(text_content)
        self.logger.info(log_msg)
        return chunks, log_msg
