import streamlit as st
import controller_llm as controller_llm
import utils as toolkit


def main():
    def set_state(i):
        st.session_state.stage = i

    # Control UI stages
    if "stage" not in st.session_state:
        st.session_state.stage = 0
    if "pdf_file_ref" not in st.session_state:
        st.session_state.pdf_file_ref = False
    if "question_text" not in st.session_state:
        st.session_state.question_text = False
    if "ctl_llm" not in st.session_state:
        st.session_state.ctl_llm = controller_llm.ControllerLlm("", "")
    if 'question_again_text' not in st.session_state:
        st.session_state.question_again_text = ''

    st.title("LLM very simple app")

    # Shows button to upload PDF
    if st.session_state.stage < 1:

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
        st.subheader("Question")
        st.write(f"LLM warmed up?: {st.session_state.ctl_llm.check_if_chain_is_ready()}")
        st.write("What do you want to know about the content in the web article?")
        text_question = st.text_input("Question:")
        if text_question:
            st.session_state.question_text = text_question
            set_state(2)

    # Shows button to get the answer
    if st.session_state.stage == 2:
        st.divider()
        st.markdown(
            f"You would ask **{st.session_state.question_text}** about the article in ```{st.session_state.url_to_ask }```."
        )
        st.button("Ask to LLM", on_click=set_state, args=[3])

    # Calls LLM and shows answer
    if st.session_state.stage == 3:
        # Using Controller LLM
        st.session_state.ctl_llm.url = st.session_state.url_to_ask
        st.session_state.ctl_llm.question = st.session_state.question_text
        st.write("Warming up LLM and then asking.")
        answer = st.session_state.ctl_llm.main()

        st.write(answer["answer"])
        st.divider()
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
        st.write("Asking with LLM warmed.")
        _, answer = st.session_state.ctl_llm.ask_question_to_llm()

        st.write(answer["answer"])
        st.divider()
        with st.expander("See details"):
            st.subheader("Details")
            st.write(answer)
            st.divider()
        st.button("Ask again", on_click=set_state, args=[4])


if __name__ == "__main__":
    # --- Library
    utils = toolkit.UtilsLLM()
    main()
