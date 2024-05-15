import streamlit as st
import controller_llm as controller_llm
import utils as toolkit
import parameters as general_parameters


def main():
    def set_state(i):
        st.session_state.stage = i

    def set_llm_model_choice():
        st.session_state.llm_model_choice = st.session_state['element_llm_model_choice']

    # Control UI stages
    if "stage" not in st.session_state:
        st.session_state.stage = 0
    if "pdf_file_ref" not in st.session_state:
        st.session_state.pdf_file_ref = False
    if "question_text" not in st.session_state:
        st.session_state.question_text = False
    if "ctl_llm" not in st.session_state:
        st.session_state.ctl_llm = controller_llm.ControllerLlm("", "")
    if "question_again_text" not in st.session_state:
        st.session_state.question_again_text = ""
    if "llm_model_choice" not in st.session_state:
        st.session_state.llm_model_choice = general_parameters.par__default_llm_model_choice

    st.title("LLM very simple app")

    # Shows button to upload PDF
    if st.session_state.stage < 1:

        st.selectbox(
            "LLM being used:",
            ("Gemini", "chatGPT"),
            on_change=set_llm_model_choice,
            key="element_llm_model_choice"
        )

        st.subheader("Article URL")
        st.write("What article do you want ask about?")
        url_to_ask = st.text_input("URL:")
        if url_to_ask:
            st.session_state.url_to_ask = url_to_ask
            set_state(1)

        # --- PDF upload
        # st.subheader("PDF file")
        # st.write("Upload a PDF file needed to answer your question:")
        # file_ref = st.file_uploader(
        #     "Upload PDF", type=["pdf"])
        # st.session_state.pdf_file_ref = file_ref

        # if st.button("Process"):
        #     if file_ref is not None:
        #         # Shows pdf file metadata
        #         file_details = {
        #             "filename": file_ref.name,
        #             "filetype": file_ref.type,
        #             "filesize": file_ref.size,
        #         }
        #         st.write(file_details)

        #         try:
        #             with pdfplumber.open(file_ref) as f:
        #                 pages = f.pages[0]
        #                 st.subheader("Page 1:")
        #                 st.write(pages.extract_text())
        #         except Exception as e:
        #             st.write(">>Error loading pdf file.")
        #             raise e

        #         set_state(1)
        #     else:
        #         st.write("Please, upload a PDF file first.")
        # ---

    # Shows input text to get the question
    if st.session_state.stage == 1:
        placeholder_question_first = st.empty()
        with placeholder_question_first.container():
            st.subheader("Question")
            st.write("What do you want to know about the content in the web article?")
            with st.form("form_question_first", border=False, clear_on_submit=True):
                text_question = st.text_input("Question:")
                submitted = st.form_submit_button("Submit")
                if submitted:
                    st.session_state.question_text = text_question
                    placeholder_question_first.empty()
                    set_state(2)

    # Shows button to get the answer
    if st.session_state.stage == 2:
        placeholder_confirmation = st.empty()
        with placeholder_confirmation.container():
            st.markdown(
                f"You would ask **{st.session_state.question_text}** about the article in "
                f"```{st.session_state.url_to_ask}```. "
            )
            st.button("Ask to LLM", on_click=set_state, args=[3])

    # Calls LLM and shows answer
    if st.session_state.stage == 3:
        # Using Controller LLM
        st.session_state.ctl_llm.url = st.session_state.url_to_ask
        st.session_state.ctl_llm.question = st.session_state.question_text

        placeholder_log = st.empty()
        with placeholder_log.container():
            st.write("Warming up LLM and then asking...")
            _ = st.session_state.ctl_llm.main()

            st.markdown(
                f"```It will take some time because first we need to warm up the LLM and their friends."
                f" If you are curious I can show you each step happening. But I will be quick so chop-chop."
                f" Enjoy the ride...```"
            )
            st.markdown(f"```Starting...```")

            # --- Authentication
            _, log_msg = st.session_state.ctl_llm.authenticate()
            st.markdown(f"```{log_msg}```")

            # Get text content
            text_content, log_msg = st.session_state.ctl_llm.get_content(
                mode="text_no_parse"
            )
            st.markdown(f"```{log_msg}```")
            if not text_content:
                st.markdown(
                    ":red[This article is not accessible by me.] Sorry. Please try another article."
                    " The app will restart soon."
                )
                set_state(0)
                utils.wait_for(
                    seconds_to_wait=general_parameters.par__waiting_time_in_seconds_in_error_case
                )
                st.experimental_rerun()

            _, log_msg = st.session_state.ctl_llm.describe_chunks(text_content)
            st.markdown(f"```{log_msg}```")

            # Define embedding model
            embedding_model, log_msg = st.session_state.ctl_llm.define_embedding_model()
            st.markdown(f"```{log_msg}```")

            # Create/reset vetorstore index
            _, log_msg = st.session_state.ctl_llm.create_vector_store_index()
            st.markdown(f"```{log_msg}```")

            # Upload vectors to vetorstore
            (
                vectorstore_from_docs,
                log_msg,
            ) = st.session_state.ctl_llm.upload_vectors_to_vector_store(
                text_content, embedding_model
            )
            st.markdown(f"```{log_msg}```")

            # Wait some time, to have vectorstore available
            st.markdown(
                f"```Waiting {general_parameters.par__waiting_time_in_seconds} seconds...```"
            )
            utils.wait_for(
                seconds_to_wait=general_parameters.par__waiting_time_in_seconds
            )
            st.markdown(f"```...continuing now.```")

            # Check if the new index exists
            _, log_msg = st.session_state.ctl_llm.check_if_vector_store_index_exists()
            st.markdown(f"```{log_msg}```")

            # Check availability of the vectorestore
            # ToDo

            # Define LLM model
            llm_model, log_msg = st.session_state.ctl_llm.define_llm_model(st.session_state.llm_model_choice)
            st.markdown(f"```{log_msg}```")

            # Prepare prompt
            prompt, log_msg = st.session_state.ctl_llm.prepare_prompt()
            st.markdown(f"```{log_msg}```")

            # Build chain
            _, log_msg = st.session_state.ctl_llm.build_chain(
                vectorstore_from_docs, llm_model, prompt, st.session_state.llm_model_choice
            )
            st.markdown(f"```{log_msg}```")

            # Ask question about the content
            _, answer = st.session_state.ctl_llm.ask_question_to_llm(st.session_state.llm_model_choice)
            st.markdown(f"```{log_msg}```")

            placeholder_log.empty()

        # Present answer
        st.write(answer["answer"])
        with st.expander("See details"):
            st.subheader("Details")
            st.write(answer)
            st.divider()
        st.button("Ask again", on_click=set_state, args=[4])

    # Shows input text to get the question, ask again
    if st.session_state.stage == 4:
        placeholder = st.empty()
        with placeholder.container():
            st.subheader("Question")
            st.write("What more do you want to know about the article?")
            with st.form("form_question_again", border=False, clear_on_submit=True):
                text_question = st.text_input("Question:")
                submitted = st.form_submit_button("Submit")
                if submitted:
                    st.session_state.question_text = text_question
                    placeholder.empty()
                    set_state(5)

    # Calls LLM and shows answer
    if st.session_state.stage == 5:
        st.session_state.ctl_llm.question = st.session_state.question_text
        placeholder_running_again = st.empty()
        with placeholder_running_again.container():
            st.markdown("```Asking with LLM warmed... now is faster.```")
            _, answer = st.session_state.ctl_llm.ask_question_to_llm(st.session_state.llm_model_choice)
        placeholder_running_again.empty()

        st.write(answer["answer"])
        with st.expander("See details"):
            st.subheader("Details")
            st.write(answer)
            st.divider()
        st.button("Ask again", on_click=set_state, args=[4])


if __name__ == "__main__":
    # --- Library
    utils = toolkit.UtilsLLM()
    main()
