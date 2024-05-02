import streamlit as st
import pdfplumber
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

    st.title("LLM very simple app")

    # Shows input text to get the question
    if st.session_state.stage <= 1:
        st.subheader("Question")
        st.write("What do you want to know about the content of a PDF file?")
        text_question = st.text_input("Question:", on_change=set_state, args=[1])
        st.session_state.question_text = text_question

    # Shows button to upload PDF
    if st.session_state.stage == 1:
        st.subheader("PDF file")
        st.write("Upload a PDF file needed to answer your question:")
        file_ref = st.file_uploader(
            "Upload PDF", type=["pdf"], on_change=set_state, args=[1]
        )
        st.session_state.pdf_file_ref = file_ref

        if st.button("Process", on_click=set_state, args=[1]):
            if file_ref is not None:
                # Shows pdf file metadata
                file_details = {
                    "filename": file_ref.name,
                    "filetype": file_ref.type,
                    "filesize": file_ref.size,
                }
                st.write(file_details)

                try:
                    with pdfplumber.open(file_ref) as f:
                        pages = f.pages[0]
                        st.subheader("Page 1:")
                        st.write(pages.extract_text())
                except Exception as e:
                    st.write(">>Error loading pdf file.")
                    raise e

                set_state(2)
            else:
                st.write("Please, upload a PDF file first.")

    # Shows button to get the answer
    if st.session_state.stage == 2:
        st.divider()
        st.markdown(
            f"You would ask **{st.session_state.question_text}** about the PDF file ```{st.session_state.pdf_file_ref.name}```."
        )
        st.button("Ask to LLM", on_click=set_state, args=[3])

    # Calls LLM
    if st.session_state.stage == 3:
        # Using Controller LLM
        ctl_llm = controller_llm.ControllerLLM_PDF_question(
            st.session_state.pdf_file_ref, st.session_state.question_text
        )
        answer = ctl_llm.main()
        st.write(answer["answer"])
        st.divider()
        st.subheader("Details")
        st.write(answer)


if __name__ == "__main__":
    # --- Library
    utils = toolkit.UtilsLLM()
    main()
