import pdfplumber
import hashlib
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

    def get_text_from_pdf_pdfplumber(self, file_path):
        """ Read a pdf file and get its content using pdf plumber"""

        with pdfplumber.open(file_path) as f:
            number_of_pages = len(f.pages)
            text_content = ""
            for i_page in range(number_of_pages):
                page = f.pages[i_page]
                text_content += page.extract_text()
            chunks, log_msg = self.split_text_into_chunks(text_content)
        return chunks, log_msg

    def define_embedding_model_google(self):
        """ Define a google embedding model to be used to transform the content in vectors """

        try:
            embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        except Exception as e:
            raise e

        log_msg = "Succeed defining the embedding model"
        self.logger.info(log_msg)
        return embedding_model, log_msg

    def check_if_a_specific_index_exists_in_pinecone(
        self, pinecone_client, par__vector_store_index_name
    ):
        """ Check if a specific index name exists in Pinecone """

        self.logger.debug(f"Looking for {par__vector_store_index_name}")
        self.logger.debug(f"... in {pinecone_client.list_indexes().names()}")
        if par__vector_store_index_name in pinecone_client.list_indexes().names():
            log_msg = "Specific index name exists in Pinecone"
            self.logger.info(log_msg)
            return True
        log_msg = "Specific index name not found in Pinecone"
        self.logger.info(log_msg)
        return False

    def check_if_any_index_exists_in_pinecone(self, pinecone_client):
        """ Check if any index exists in Pinecone """

        total_of_indexes_found = len(pinecone_client.list_indexes().names())
        self.logger.debug(
            f"Total of indexes found in Pinecone: {total_of_indexes_found}"
        )
        if total_of_indexes_found > 0:
            log_msg = "Some index exist in Pinecone"
            self.logger.info(log_msg)
            return True
        log_msg = "No index was found in Pinecone"
        self.logger.info(log_msg)
        return False

    @staticmethod
    def list_pinecone_index_names(pinecone_client):
        """ List the indexes existing in a Pinecone project """

        return pinecone_client.list_indexes().names()

    def create_pinecone_index(self, pinecone_client, par__vector_store_index_name):
        """ Create a namespace in Pinecone database """

        # If exists some index and it is not the same that need to be created
        if self.check_if_any_index_exists_in_pinecone(
            pinecone_client
        ) and not self.check_if_a_specific_index_exists_in_pinecone(
            pinecone_client, par__vector_store_index_name
        ):
            # Delete index before create it again
            index_name_to_be_deleted = self.get_current_index_pinecone(pinecone_client)
            if not index_name_to_be_deleted:
                raise Exception(
                    "More than one index found in vector store. The expectation to find only 1 index."
                )
            try:
                pinecone_client.delete_index(index_name_to_be_deleted)
                self.logger.info(
                    f"Index {index_name_to_be_deleted} deleted of vector store."
                )
            except Exception as e:
                log_msg = "Failed trying to delete pinecone index"
                self.logger.error(log_msg)
                raise e

        # Create index
        try:
            pinecone_client.create_index(
                name=par__vector_store_index_name,
                dimension=768,
                metric="euclidean",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        except Exception as e:
            log_msg = "Failed creating pinecone index"
            self.logger.error(log_msg)
            raise e
        log_msg = "Succeed creating pinecone index"
        self.logger.info(log_msg)
        return True, log_msg

    def upload_vectors_to_pinecone(
        self, par__vector_store_index_name, chunks, embedding_model
    ):
        """ Upload vectors of embedded content to Pinecone """

        # Upload vectors to Pinecone
        try:
            vectorstore_loaded = PineconeVectorStore.from_documents(
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
        return vectorstore_loaded, log_msg

    def define_llm_model_google(self):
        """ Define the LLM model as a Google model """

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

    def define_llm_model_openai(self):
        """ Define the LLM model as a OpenAI model """

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

    def prepare_prompt_with_vector_store(self):
        """ Prepare prompt considering the use of a vector store as a RAG """

        try:
            prompt = PromptTemplate(
                template=general_parameters.par__prompt_template_generic_chain,
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

    def build_retrieved_documents_chain(
        self, vector_store_loaded_client, llm_model, prompt
    ):
        """ Build a chain to ask questions to a LLM model using documents retrieved from a vector store as input """

        try:
            retriever = vector_store_loaded_client.as_retriever()
            combine_docs_chain = create_stuff_documents_chain(llm_model, prompt)
            retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        except Exception as e:
            raise e
        log_msg = "Succeed building chain with documents retrieval"
        self.logger.info(log_msg)
        return retrieval_chain, log_msg

    def build_all_documents_chain(self, llm_model, prompt):
        """ Build a chain that passes all documents to the LLM as context """

        try:
            all_docs_chain = create_stuff_documents_chain(llm_model, prompt)
        except Exception as e:
            raise e
        log_msg = "Succeed building chain with all documents"
        self.logger.info(log_msg)
        return all_docs_chain, log_msg

    def asking_question_about_content_using_retrieved_documents_chain(
        self, retrieval_chain, question
    ):
        """ Ask a question to a LLM model using a retrieved documents chain """

        try:
            answer_about_content = retrieval_chain.invoke(
                {general_parameters.par__prompt_template_var_input: question}
            )
        except Exception as e:
            log_msg = "Failed asking question about content"
            self.logger.error(log_msg)
            raise e
        log_msg = (
            "Succeed asking question about content using a retrieved documents chain"
        )
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
        log_msg = "Text split into chunks with success."
        self.logger.info(log_msg)
        return chunks, log_msg

    def get_text_from_web_article_parsing_text(self, url):
        text_content, log_msg = self.get_text_from_web_article(url)
        chunks, log_msg = self.split_text_into_chunks(text_content)
        return chunks, log_msg

    @staticmethod
    def apply_hash_md5(string_to_hash):
        """ Apply a hash to a string """

        md5_hash = hashlib.md5()
        md5_hash.update(string_to_hash.encode("utf-8"))
        return md5_hash.hexdigest()

    def set_vector_store_client_to_specific_index(self, url, embedding_model):
        """ Return a Pinecone client set for a specific index """

        try:
            index_name_from_url_hashed = self.apply_hash_md5(url)
            vectorstore_loaded = PineconeVectorStore.from_existing_index(
                index_name=index_name_from_url_hashed, embedding=embedding_model
            )
        except Exception as e:
            log_msg = "Failed setting Pinecone client to a specific index"
            self.logger.error(log_msg)
            raise e
        log_msg = "Succeed setting Pinecone client to a specific index"
        self.logger.info(log_msg)
        return vectorstore_loaded

    def get_current_index_pinecone(self, pinecone_client):
        """ Get the current unique index set in pinecone """

        indexes_found = pinecone_client.list_indexes().names()
        if len(indexes_found) > 1:
            self.logger.error(
                "More than 1 index found in vector store. Not clear which index to return."
            )
            return None
        elif len(indexes_found) < 1:
            self.logger.error("No index was found in vector store. No index to return.")
            return None
        else:
            self.logger.debug(f"Index found and returned: {indexes_found[0]}.")
            return indexes_found[0]

    def describe_chunks(self, chunks):
        """ Check and print metrics about the text chunks """

        content = "\n".join(str(p.page_content) for p in chunks)
        log_msg = ""
        log_msg += f"Total of characters in the content: {len(content)}"
        log_msg += f", and total of chunks: {len(chunks)}"
        self.logger.info(log_msg)
        return None, log_msg

    def split_documents_into_chunks(
        self, documents_content, parr_chunk_size=500, parr_chunk_overlap=50
    ):
        """ Split text into chunks """

        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=parr_chunk_size, chunk_overlap=parr_chunk_overlap
            )
            chunks = text_splitter.split_documents(documents_content)
            log_msg = "Documents splitted into chunks with success."
            self.logger.info(log_msg)
            return chunks, log_msg
        except Exception as e:
            raise e

    def asking_question_about_content_using_all_documents_chain(
        self, all_documents_chain, question, limited_text_documents
    ):
        """ Ask a question to a LLM model using a chain with all documents """

        try:
            answer_about_content = all_documents_chain.invoke(
                {
                    general_parameters.par__prompt_template_var_context: limited_text_documents,
                    general_parameters.par__prompt_template_var_input: question,
                }
            )

            # ToDo: allow a debug log with prompt fulfilled for checking

        except Exception as e:
            log_msg = "Failed asking question about content"
            self.logger.error(log_msg)
            raise e
        log_msg = "Succeed asking question about content using a all documents chain"
        self.logger.info(log_msg)
        return answer_about_content, log_msg
