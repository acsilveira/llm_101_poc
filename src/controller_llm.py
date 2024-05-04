import google.generativeai as genai
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
import utils as toolkit
import parameters as general_parameters


class ControllerLlm:
    def __init__(self, url, question):
        self._url = url
        self._question = question
        self._utils = toolkit.UtilsLLM()
        self._vector_store_client = None
        self._chain = None
        self._verbose_mode = True
        self._chain_ready = False

    @property
    def url(self):
        return self._url

    @property
    def question(self):
        return self._question

    @property
    def utils(self):
        return self._utils

    @property
    def vector_store_client(self):
        return self._vector_store_client

    @property
    def chain(self):
        return self._chain

    @property
    def verbose_mode(self):
        return self._verbose_mode

    @property
    def chain_ready(self):
        return self._chain_ready

    @url.setter
    def url(self, value):
        self._url = value

    @question.setter
    def question(self, value):
        self._question = value

    @utils.setter
    def utils(self, value):
        self._utils = value

    @vector_store_client.setter
    def vector_store_client(self, value):
        self._vector_store_client = value

    @chain.setter
    def chain(self, value):
        self._chain = value

    @verbose_mode.setter
    def verbose_mode(self, value):
        self._verbose_mode = value

    @chain_ready.setter
    def chain_ready(self, value):
        self._chain_ready = value

    def main(self):
        # self._utils.log("Starting...")

        # # --- Authentication
        # _, log_msg = self.authenticate()
        # self._utils.log(log_msg) if self._verbose_mode else None
        #
        # # Get text content
        # text_content, log_msg = self.get_content()
        # self._utils.log(log_msg) if self._verbose_mode else None

        # # Split content in chunks
        # # chunks, log_msg = self.split_text_into_chunks(text_content)
        # # chunks, log_msg = self.split_documents_into_chunks(text_content)
        # chunks, log_msg = text_content, "Using chunks only by HTML header"
        # self._utils.log(log_msg)
        # content = "\n".join(str(p.page_content) for p in chunks)
        # self._utils.log(f"The total words in the content is: {len(content)}")
        # self._utils.log(f"The total chunks in the content is: {len(chunks)}")

        # # Define the embedding model
        # embedding_model, log_msg = self.define_embedding_model()
        # self._utils.log(log_msg)

        # # Create/reset vetorstore index
        # _, log_msg = self._utils.create_pinecone_index(
        #     self._vector_store_client, general_parameters.par__vector_store_index_name
        # )
        # self._utils.log(log_msg)

        # # Upload vectors to vetorstore
        # vectorstore_from_docs, log_msg = self._utils.upload_vectors_to_vectorstore(
        #     self._vector_store_client, general_parameters.par__vector_store_index_name, chunks, embedding_model
        # )
        # self._utils.log(log_msg)

        # # Wait some time, to have vectorstore available
        # self._utils.wait_for(seconds_to_wait=5)

        # Check availability of the vectorestore
        #ToDo

        # # Check if the new index exists
        # _, log_msg = self._utils.check_if_pinecone_index_exists(self._vector_store_client, general_parameters.par__vector_store_index_name)
        # self.utils.log(log_msg)

        # # Test retrieval from embeedings
        # query = "disability"
        # result = vectorstore_from_docs.similarity_search(query)
        # print(result)

        # # Define LLM model
        # llm_model, log_msg = self._utils.define_llm_model()
        # self._utils.log(log_msg)

        # # Prepare prompt
        # prompt, log_msg = self._utils.prepare_prompt()
        # self._utils.log(log_msg)

        # # Build chain wo retriever
        # chain, log_msg = self.utils.build_chain_woRetriver(llm_model, prompt, general_parameters.par__prompt_template_var_context, general_parameters.par__prompt_template_var_input, text_content, self.question)
        # self.utils.log(log_msg)

        # # Build chain
        # self._chain, log_msg = self._utils.build_chain(vectorstore_from_docs, llm_model, prompt)
        # self.chain_ready = True
        # self._utils.log(log_msg)

        # # Ask question about the content
        # _, answer = self.ask_question_to_llm()
        # answer, log_msg = self._utils.asking_question_about_content(self._chain, self._question)
        # self._utils.log("Question asked and answer received.")
        pass

    def split_text_into_chunks(self, pdf_raw_text_content):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0,)
        chunks = text_splitter.create_documents([pdf_raw_text_content])

        return chunks, "Text splitted into chunks with success."

    def split_documents_into_chunks(self, documents_content, parr_chunk_size=500, parr_chunk_overlap=50):
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=parr_chunk_size, chunk_overlap=parr_chunk_overlap)
            chunks = text_splitter.split_documents(documents_content)
            return chunks, "Documents splitted into chunks with success."
        except Exception as e:
            raise e

    def authenticate(self):
        """ Do the authentications needed """

        try:
            self._utils.log(load_dotenv())  # Check env_vars
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))  # Auth Google
            self._vector_store_client = Pinecone(
                api_key=str(os.getenv("PINECONE_API_KEY")).strip('"')
            )  # Auth Pinecone
            log_msg = "Succeed authenticating."

            return True, log_msg
        except Exception as e:
            raise e

    def ask_question_to_llm(self):
        answer, log_msg = self._utils.asking_question_about_content(self._chain, self._question)
        self._utils.log(log_msg)
        self._utils.log(f"Q: {self._question}")
        self._utils.log("A: " + answer["answer"])
        print(answer)
        return True, answer

    def check_if_chain_is_ready(self):
        return self._chain_ready

    def get_content(self):
        text_content, log_msg = self._utils.get_text_from_web_article_parsing_htmlLangChain(self.url)
        return text_content, log_msg

    def describe_chunks(self, chunks):
        content = "\n".join(str(p.page_content) for p in chunks)
        log_msg = ""
        log_msg += f"Total of words in the content: {len(content)}"
        log_msg += f", and total of chunks: {len(chunks)}"
        return None, log_msg

    def define_embedding_model(self):
        embedding_model, log_msg = self._utils.define_embedding_model()
        return embedding_model, log_msg

    def create_vector_store_index(self):
        _, log_msg = self._utils.create_pinecone_index(
            self._vector_store_client, general_parameters.par__vector_store_index_name
        )
        return None, log_msg

    def upload_vectors_to_vector_store(self, chunks, embedding_model):
        vectorstore_from_docs, log_msg = self._utils.upload_vectors_to_vectorstore(
            self._vector_store_client, general_parameters.par__vector_store_index_name, chunks, embedding_model
        )
        return vectorstore_from_docs, log_msg

    def check_if_vector_store_index_exists(self):
        _, log_msg = self._utils.check_if_pinecone_index_exists(
            self._vector_store_client,
            general_parameters.par__vector_store_index_name
        )
        return _, log_msg

    def define_llm_model(self):
        llm_model, log_msg = self._utils.define_llm_model()
        return llm_model, log_msg

    def prepare_prompt(self):
        prompt, log_msg = self._utils.prepare_prompt()
        return prompt, log_msg

    def build_chain(self, vectorstore_from_docs, llm_model, prompt):
        self._chain, log_msg = self._utils.build_chain(vectorstore_from_docs, llm_model, prompt)
        self.chain_ready = True
        return None, log_msg