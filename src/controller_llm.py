import google.generativeai as genai
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from pprint import pprint
import utils as toolkit
import time


class ControllerLLM_PDF_question:
    def __init__(self, pdf_file_ref, question):
        self.pdf_file_ref = pdf_file_ref
        self.question = question

    def main(self):
        # --- Library
        utils = toolkit.UtilsLLM()

        # --- Parameters
        import parameters as general_parameters

        utils.log("Starting...")

        # --- Authentication
        utils.log(load_dotenv())  # Check env_vars
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))  # Auth Google
        pc = Pinecone(
            api_key=str(os.getenv("PINECONE_API_KEY")).strip('"')
        )  # Auth Pinecone

        # Get text content from PDF
        text_content = utils.read_pdf(self.pdf_file_ref)

        # Split content in chunks
        chunks, log_msg = self.split_text_into_chunks(text_content)
        utils.log(log_msg)
        content = "\n".join(str(p.page_content) for p in chunks)
        utils.log(f"The total words in the content is: {len(content)}")

        # Define the embedding model
        embedding_model, log_msg = utils.define_embedding_model()
        utils.log(log_msg)

        # # Test embeddings
        # query_result = embedding_model.embed_query(chunks[0].page_content)
        # utils.log(query_result)

        # Create/reset vetorstore index
        _, log_msg = utils.create_pinecone_index(
            pc, general_parameters.par__vector_store_index_name
        )
        utils.log(log_msg)
        time.sleep(5)  # Wait to upload vectors

        # Upload vectors to vetorstore
        vectorstore_from_docs, log_msg = utils.upload_vectors_to_vectorstore(
            pc, general_parameters.par__vector_store_index_name, chunks, embedding_model
        )
        utils.log(log_msg)

        # # Test retrieval from embeedings
        # query = "disability"
        # result = vectorstore_from_docs.similarity_search(query)
        # print(result)

        # Define LLM model
        llm_model, log_msg = utils.define_llm_model()
        utils.log(log_msg)

        # Prepare prompt
        prompt, log_msg = utils.prepare_prompt()
        utils.log(log_msg)

        # Build chain
        chain, log_msg = utils.build_chain(vectorstore_from_docs, llm_model, prompt)
        utils.log(log_msg)

        # Ask question about the content
        answer, log_msg = utils.asking_question_about_content(chain, self.question)
        utils.log(log_msg)
        utils.log(f"Q: {self.question}")
        utils.log("A: " + answer["answer"])
        print(answer)

        utils.log("End.")
        return answer

    def split_text_into_chunks(self, pdf_raw_text_content):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0,)
        chunks = text_splitter.create_documents([pdf_raw_text_content])

        return chunks, "Text splitted into chunks with success."


if __name__ == "__main__":
    ctr_llm = ControllerLLM_PDF_question(
        "../data/articleAccessibleDesign.pdf"
        # ,"How to support hearing disability?"
        ,
        "What is the benefit of designing accessible products?",
    )
    ctr_llm.main()
