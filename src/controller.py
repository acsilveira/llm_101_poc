import google.generativeai as genai
import os
from dotenv import load_dotenv
from pinecone import Pinecone
import utils as toolkit
import parameters as general_parameters
from logging_setter import logger


class Controller:
    def __init__(self, content_ref, question):
        self._content_ref = content_ref
        self._pdf_file_ref = None
        self._content_type = None
        self._question = question
        self._model_choice = None
        self._utils = toolkit.UtilsLLM()
        self._vector_store_client = None
        self._vector_store_loaded_client = None
        self._embedding_model = None
        self._chain = None
        self._verbose_mode = True
        self._chain_ready = False
        self.logger = logger
        self.logger.info("Controller initialized")
        self._is_authenticated = False
        self._chain_is_prepared = False

    @property
    def content_ref(self):
        return self._content_ref

    @property
    def pdf_file_ref(self):
        return self._pdf_file_ref

    @property
    def content_type(self):
        return self._content_type

    @property
    def question(self):
        return self._question

    @property
    def model_choice(self):
        return self._model_choice

    @property
    def utils(self):
        return self._utils

    @property
    def vector_store_client(self):
        return self._vector_store_client

    @property
    def vector_store_loaded_client(self):
        return self._vector_store_loaded_client

    @property
    def chain(self):
        return self._chain

    @property
    def verbose_mode(self):
        return self._verbose_mode

    @property
    def chain_ready(self):
        return self._chain_ready

    @property
    def embedding_model(self):
        return self._embedding_model

    @property
    def chain_is_prepared(self):
        return self._chain_is_prepared

    @content_ref.setter
    def content_ref(self, value):
        self._content_ref = value

    @pdf_file_ref.setter
    def pdf_file_ref(self, value):
        self._pdf_file_ref = value

    @content_type.setter
    def content_type(self, value):
        self._content_type = value

    @question.setter
    def question(self, value):
        self._question = value

    @model_choice.setter
    def model_choice(self, value):
        self._model_choice = value

    @utils.setter
    def utils(self, value):
        self._utils = value

    @vector_store_client.setter
    def vector_store_client(self, value):
        self._vector_store_client = value

    @vector_store_loaded_client.setter
    def vector_store_loaded_client(self, value):
        self._vector_store_loaded_client = value

    @chain.setter
    def chain(self, value):
        self._chain = value

    @verbose_mode.setter
    def verbose_mode(self, value):
        self._verbose_mode = value

    @chain_ready.setter
    def chain_ready(self, value):
        self._chain_ready = value

    @embedding_model.setter
    def embedding_model(self, value):
        self._embedding_model = value

    @chain_is_prepared.setter
    def chain_is_prepared(self, value):
        self._chain_is_prepared = value

    def ask_to_llm(
        self,
        content_to_ask,
        question_to_ask,
        model_to_ask,
        content_type_to_ask,
        pdf_file_ref,
    ):
        """ Ask a question about a content to a LLM model """

        self.logger.info(
            "\nIt will take some time because first we need to warm up the LLM and their friends."
            "\nIf you are curious I can show you each step happening. But I will be quick so chop-chop."
            "\nEnjoy the ride."
            "\nStarting..."
        )

        # Authenticate
        if not self._is_authenticated:
            _, log_msg = self.authenticate()
        else:
            self.logger.info("Authentication is already set.")

        # Define embedding model
        _, _ = self.define_embedding_model()

        # Register the content type chosen
        self.content_type = content_type_to_ask

        # Prepare content
        if not self.check_if_vector_store_index_already_exists_for_this_url(
            content_to_ask
        ):
            self.logger.info(
                "Once the URL-based index was not found, preparing the vector store for a new content"
            )
            self.content_ref = content_to_ask
            self.pdf_file_ref = pdf_file_ref
            (
                return_success,
                log_msg,
                self.vector_store_loaded_client,
            ) = self.prepare_content_in_vector_store(content_type_to_ask)
            if return_success < 0:
                return return_success, log_msg
        else:
            self.logger.info(
                "Vector store already fulfilled with this URL content. Setting client to respective index..."
            )
            if not self.content_ref:
                self.content_ref = content_to_ask
            self.vector_store_loaded_client = self.utils.set_vector_store_client_to_specific_index(
                self.content_ref, self.embedding_model
            )

        # Ask question about the content
        self.question = question_to_ask
        _, answer = self.ask_question_to_llm(model_to_ask, content_type_to_ask)
        return 1, answer

    def authenticate(self):
        """ Authenticate APIs """

        try:
            self.logger.info(f"Env vars loaded: {load_dotenv()}")  # Check env_vars
            self.logger.info(
                f"GOOGLE_API_KEY len: {len(os.environ.get('GOOGLE_API_KEY'))}"
            )
            self.logger.info(
                f"PINECONE_API_KEY len: {len(os.environ.get('PINECONE_API_KEY'))}"
            )
            self.logger.info(
                f"OPENAI_API_KEY len: {len(os.environ.get('OPENAI_API_KEY'))}"
            )
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))  # Auth Google
            self._vector_store_client = Pinecone(
                api_key=str(os.getenv("PINECONE_API_KEY")).strip('"')
            )  # Auth Pinecone
            log_msg = "Succeed authenticating."
            self.logger.info(log_msg)
            self._is_authenticated = True
            return True, log_msg
        except Exception as e:
            raise e

    def define_embedding_model(self):
        """ Define the embedding model to be used to transform the content in vectors """

        embedding_model, log_msg = self._utils.define_embedding_model_google()
        self.embedding_model = embedding_model
        self.logger.debug(f"Embedding model defined: {self.embedding_model}")
        return embedding_model, log_msg

    def prepare_content_in_vector_store(self, content_type):
        """ Get content, apply embedding and save vectors in a vector store """

        self.logger.info("Preparing vector store...")
        # Get text content
        if content_type == "URL":
            text_content, log_msg = self.get_content(mode="text_no_parse")
        elif content_type == "PDF":
            text_content, log_msg = self.get_content(mode="pdf_text")
        else:
            return -3, "Unknown type of content", None
        if not text_content:
            return -1, "Content not accessible", None

        _, log_msg = self.utils.describe_chunks(text_content)

        # Create/reset vector store index
        _, log_msg = self.create_vector_store_index()

        # Upload vectors to vector store
        (vector_store_loaded_client, log_msg,) = self.upload_vectors_to_vector_store(
            text_content, self.embedding_model
        )

        # Wait some time, to have vectorstore available
        self._utils.wait_for(
            seconds_to_wait=general_parameters.par__waiting_time_in_seconds
        )

        # Check if the new index exists
        result_success = self.check_if_vector_store_index_already_exists_for_this_url(
            self.content_ref
        )
        if not result_success:
            return -2, "Namespace does not exist in vector store", None

        # ToDo: Check availability of the vectorestore

        self.logger.info("Preparing vector store...DONE.")
        return 1, log_msg, vector_store_loaded_client

    def ask_question_to_llm(self, llm_model_choice, content_type_to_ask):
        """ Ask a question to a LLM model """

        # Prepare LLM chain
        self.logger.debug(f"Flag chain prepared set as {self._chain_is_prepared}")
        self.logger.debug(
            f"Model choice previously set as {self.model_choice} and passed as {llm_model_choice}"
        )
        self.logger.debug(
            f"Content type choice previously set as {self.content_type} and passed as {content_type_to_ask}"
        )
        if (
            not self._chain_is_prepared
            or self.model_choice != llm_model_choice
            or self.content_type != content_type_to_ask
        ):
            self.model_choice = llm_model_choice
            self.prepare_llm_chain(self.vector_store_loaded_client)
            self.logger.info(f"Succeed resetting chain for model: {llm_model_choice}.")
        else:
            self.logger.info("Chain is already set. Reusing it.")

        # Run LLM chain
        (
            answer,
            log_msg,
        ) = self._utils.asking_question_about_content_using_retrieved_documents_chain(
            self._chain, self._question
        )
        self.logger.info(f"Question asked to {llm_model_choice}")
        self.logger.info(f"Q: {self._question}")
        self.logger.info("A: " + answer["answer"])
        return True, answer

    def get_content(self, mode):
        """ Get text content from a source """

        if mode == "html_parse":
            (
                text_content,
                log_msg,
            ) = self._utils.get_text_from_web_article_parsing_html_langchain(
                self.content_ref
            )
        elif mode == "text_no_parse":
            (
                text_content,
                log_msg,
            ) = self._utils.get_text_from_web_article_parsing_text(self.content_ref)
        elif mode == "pdf_text":
            (text_content, log_msg,) = self._utils.get_text_from_pdf_pdfplumber(
                self.pdf_file_ref
            )
        else:
            self.logger.error("Unknown mode to get content")
            return None, "Unknown mode to get content."
        return text_content, log_msg

    def create_vector_store_index(self):
        """ Create a namespace in a vector database """

        # Prepare index name
        hashed_index_name = self._utils.apply_hash_md5(self.content_ref)

        _, log_msg = self._utils.create_pinecone_index(
            self._vector_store_client, hashed_index_name
        )
        return None, log_msg

    def upload_vectors_to_vector_store(self, chunks, embedding_model):
        """ Upload vectors of the content embedded to a vector database """

        vectorstore_loaded, log_msg = self._utils.upload_vectors_to_pinecone(
            self.utils.apply_hash_md5(self.content_ref), chunks, embedding_model,
        )
        return vectorstore_loaded, log_msg

    def check_if_vector_store_index_already_exists_for_this_url(self, url_to_check):
        """ Check if the specific URL-based namespace already exists in a vector store """

        url_to_check_hashed = self.utils.apply_hash_md5(url_to_check)
        self.logger.debug(
            f"Checking URL {url_to_check} hashed to {url_to_check_hashed}"
        )
        result_success = self._utils.check_if_a_specific_index_exists_in_pinecone(
            self._vector_store_client, url_to_check_hashed
        )
        self.logger.info(
            f"Does a vector store index already exist for this URL: {result_success}"
        )
        return result_success

    def prepare_llm_chain(self, vector_store_loaded_client):
        """ Prepare LLM model, prompt and chain """

        # Define LLM model
        llm_model, log_msg = self.define_llm_model(self.model_choice)

        # Prepare prompt
        prompt, log_msg = self.prepare_prompt()

        # Build chain
        _, log_msg = self.build_chain(vector_store_loaded_client, llm_model, prompt)
        self._chain_is_prepared = True
        pass

    def define_llm_model(self, llm_model_choice):
        """ Define a llm model to be asked """

        if llm_model_choice == "Gemini":
            llm_model, log_msg = self._utils.define_llm_model_google()
        elif llm_model_choice == "chatGPT":
            llm_model, log_msg = self._utils.define_llm_model_openai()
        else:
            raise Exception("Unknown model choice")
        return llm_model, log_msg

    def prepare_prompt(self):
        """ Prepare the prompt for asking to LLM """

        prompt, log_msg = self._utils.prepare_prompt_with_vector_store()
        return prompt, log_msg

    def build_chain(self, vector_store_loaded_client, llm_model, prompt):
        """ Build the LLM chain to ask a question to the LLM model """

        self._chain, log_msg = self._utils.build_retrieved_documents_chain(
            vector_store_loaded_client, llm_model, prompt
        )
        self.chain_ready = True
        return None, log_msg
