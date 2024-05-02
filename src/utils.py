import textwrap
from IPython.display import Markdown
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
import parameters as general_parameters
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import PromptTemplate


class UtilsLLM:
    """
    Set of useful transformations to handle LLM interactions.
    """

    def to_markdown(self, text):
        """ Interpret string as markdown """
        text = text.replace("•", "  *")
        return Markdown(textwrap.indent(text, "> ", predicate=lambda _: True))

    def get_text_from_web_article(self, url):
        """ Get text from a web URL article """

        try:
            response = requests.get(url)
            # Check if request was successful
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                text = soup.get_text()
                parsed_text = text.replace("\n", " ")
                return parsed_text
            else:
                print("Failed to fetch content from URL:", url)
                return None
        except Exception as e:
            print("An error occurred:", e)
            return None

    def read_pdf(self, file_path):
        """ Read a pdf file and get its content """

        pdf_reader = PdfReader(file_path)
        file_text_content = pdf_reader
        text_content = ""
        number_of_pages = len(pdf_reader.pages)
        for i_page in range(number_of_pages):
            page = pdf_reader.pages[i_page]
            text_content += page.extract_text()
        return text_content

    def log(self, text):
        """
        Log text along code
            ToDo: Implement real logging
        """

        print(f">>{text}")
        pass

    def define_embedding_model(self):
        try:
            embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        except Exception as e:
            raise e
            return None, "Failed trying the define the embedding model."

        return embedding_model, "Succeed defining the embedding model"

    def create_pinecone_index(self, pinecone_client, par__vector_store_index_name):

        # If index already exists
        if par__vector_store_index_name in pinecone_client.list_indexes().names():
            # Delete index before create it again
            try:
                pinecone_client.delete_index(par__vector_store_index_name)
            except Exception as e:
                raise e
                return None, "Failed trying to delete pinecone index"

        # Create index
        try:
            pinecone_client.create_index(
                name=par__vector_store_index_name,
                dimension=768,  # Replace with your model dimensions
                metric="euclidean",  # Replace with your model metric
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        except Exception as e:
            raise e
            return None, "Failed creating pinecone index"

        return True, "Succeed creating pinecone index"

    def upload_vectors_to_vectorstore(
        self, pinecone_client, par__vector_store_index_name, chunks, embedding_model
    ):
        # Upload vectors to Pinecone
        try:
            vectorstore_from_docs = PineconeVectorStore.from_documents(
                chunks,
                index_name=par__vector_store_index_name,
                embedding=embedding_model,
            )
        except Exception as e:
            raise e
            return None, "Failed uploading vectors to vectorstore"

        return vectorstore_from_docs, "Succeed uploading vectors to vectorstore"

    def define_llm_model(self):
        try:
            llm_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        except Exception as e:
            raise e
            return None, "Failed trying the define the llm model."

        return llm_model, "Succeed defining the llm model"

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
            raise e
            return None, "Failed preparing prompt"

        return prompt, "Succeed preparing prompt"

    def build_chain(self, vectorstore_from_docs, llm_model, prompt):
        try:
            retriever = vectorstore_from_docs.as_retriever()

            combine_docs_chain = create_stuff_documents_chain(llm_model, prompt)

            retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        except Exception as e:
            raise e
            return None, "Failed building chain"

        return retrieval_chain, "Succeed building chain"

    def asking_question_about_content(self, retrieval_chain, question):
        try:
            answer_about_content = retrieval_chain.invoke(
                {general_parameters.par__prompt_template_var_input: question}
            )
        except Exception as e:
            raise e
            return None, "Failed asking question about content"

        return answer_about_content, "Succeed asking question about content"
