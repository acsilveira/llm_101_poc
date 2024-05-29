import streamlit as st
import controller as controller_llm
import utils as toolkit
import parameters as general_parameters
import pdfplumber
import logging
from custom_log_handler import StreamlitLogHandler


def setup_logging():
    root_logger = logging.getLogger()  # Get the root logger
    log_container = st.container()  # Create a container within which we display logs
    handler = StreamlitLogHandler(log_container)
    handler.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    return handler


def main():
    def set_state(i):
        st.session_state.stage = i

    def set_llm_model_choice():
        st.session_state.llm_model_choice = st.session_state["element_llm_model_choice"]

    def set_source_type_choice():
        st.session_state.source_type_choice = st.session_state[
            "element_source_type_choice"
        ]

    def set_text_handling_choice():
        st.session_state.text_handling_choice = st.session_state["element_text_handling_choice"]

    # Control UI stages
    if "stage" not in st.session_state:
        st.session_state.stage = 0
    if "content_ref_to_ask" not in st.session_state:
        st.session_state.content_ref_to_ask = False
    if "pdf_file_ref" not in st.session_state:
        st.session_state.pdf_file_ref = False
    if "question_text" not in st.session_state:
        st.session_state.question_text = False
    if "ctl_llm" not in st.session_state:
        st.session_state.ctl_llm = controller_llm.Controller("", "")
    if "question_again_text" not in st.session_state:
        st.session_state.question_again_text = ""
    if "llm_model_choice" not in st.session_state:
        st.session_state.llm_model_choice = (
            general_parameters.par__default_llm_model_choice
        )
    if "text_handling_choice" not in st.session_state:
        st.session_state.text_handling_choice = (
            general_parameters.par__default_text_handling_choice
        )
    if "source_type_choice" not in st.session_state:
        st.session_state.source_type_choice = (
            general_parameters.par__default_source_type_choice
        )

    st.title("LLM very simple app")

    placeholder_choice = st.empty()
    with placeholder_choice.container():
        if st.session_state.stage not in general_parameters.par__stages_when_choices_are_disabled:
            st.radio(
                "LLM being used",
                ["Gemini", "chatGPT"],
                on_change=set_llm_model_choice,
                key="element_llm_model_choice",
            )
            st.radio(
                "Source type",
                ["URL", "PDF"],
                on_change=set_source_type_choice,
                key="element_source_type_choice",
            )
            st.radio(
                "Content handling",
                ["Filter relevant parts", "All text"],
                on_change=set_text_handling_choice,
                key="element_text_handling_choice",
            )

    placeholder_general = st.empty()
    # --------------------------------------------------
    # Screen: Get content source
    # --------------------------------------------------
    if st.session_state.stage < 1:
        # placeholder_url = st.empty()
        with placeholder_general.container():
            if st.session_state.source_type_choice == "URL":
                st.subheader("Article URL")
                st.write("What article do you want ask about?")
                with st.form("form_content_url", border=False, clear_on_submit=True):
                    url_to_ask = st.text_input("URL:")
                    submitted = st.form_submit_button("Submit")
                    if submitted:
                        if not url_to_ask:
                            url_to_ask = (
                                general_parameters.par__default_url_content_to_test
                            )
                        st.session_state.content_ref_to_ask = url_to_ask
                        placeholder_general.empty()
                        set_state(1)
            elif st.session_state.source_type_choice == "PDF":
                st.subheader("PDF file")
                st.write("Upload a PDF file to ask about:")
                file_ref = st.file_uploader("Upload PDF", type=["pdf"])
                if st.button("Process"):
                    if file_ref is not None:
                        st.session_state.content_ref_to_ask = file_ref.name
                        st.session_state.pdf_file_ref = file_ref
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
                        set_state(1)
                    else:
                        st.write("Please, upload a PDF file first.")
            else:
                st.write(":red[Unknown content type.]")

    # --------------------------------------------------
    # Screen: Get question to be asked
    # --------------------------------------------------
    if st.session_state.stage == 1:
        # placeholder_question_first = st.empty()
        placeholder_general.empty()
        with placeholder_general.container():
            st.subheader("Question")
            st.write("What do you want to know about the content in the web article?")
            with st.form("form_question_first", border=False, clear_on_submit=True):
                text_question = st.text_input("Question:")
                submitted = st.form_submit_button("Submit")
                if submitted:
                    st.session_state.question_text = text_question
                    # placeholder_question_first.empty()
                    set_state(2)
            st.button("Star over", on_click=set_state, args=[0])

    # --------------------------------------------------
    # Screen: Confirm source and question
    # --------------------------------------------------
    if st.session_state.stage == 2:
        # if "placeholder_question_first" in st.session_state:
        #     placeholder_question_first.empty()
        # placeholder_confirmation = st.empty()
        placeholder_general.empty()
        with placeholder_general.container():
            st.markdown(
                f"You would ask **{st.session_state.question_text}** about the article in "
                f"```{st.session_state.content_ref_to_ask}```. "
            )
            st.button("Ask to LLM", on_click=set_state, args=[3])

    # --------------------------------------------------
    # Screen: Calls LLM and shows answer
    # --------------------------------------------------
    if st.session_state.stage == 3:
        handler = setup_logging()
        placeholder_choice.empty()
        placeholder_log = st.empty()
        with placeholder_log.container():
            st.write("Warming up LLM and then asking...")

            result_success, answer = st.session_state.ctl_llm.ask_to_llm(
                st.session_state.content_ref_to_ask,
                st.session_state.question_text,
                st.session_state.llm_model_choice,
                st.session_state.source_type_choice,
                st.session_state.pdf_file_ref,
            )

            if result_success == -1:
                st.markdown(
                    ":red[This article is not accessible by me.] Sorry. Please try another article."
                    " The app will restart soon."
                )
                set_state(0)
                utils.wait_for(
                    seconds_to_wait=general_parameters.par__waiting_time_in_seconds_in_error_case
                )
                st.experimental_rerun()
            elif result_success == -2:
                st.markdown(
                    ":red[The namespace was not found in the vector store.] Sorry. Please check the connection"
                    " with the vector store."
                    " The app will restart soon."
                )
                set_state(0)
                utils.wait_for(
                    seconds_to_wait=general_parameters.par__waiting_time_in_seconds_in_error_case
                )
                st.experimental_rerun()
            elif result_success == -3:
                st.markdown(":red[Unknown content type.] Sorry. This is not expected.")
                raise Exception("Unknown content type to get content from.")

            st.markdown("Success asking question to the LLM model.")
            placeholder_log.empty()

        # Present answer
        st.write(answer["answer"])
        with st.expander("See details"):
            st.subheader("Details")
            st.write(answer)
            st.divider()
        st.button("Ask again", on_click=set_state, args=[4])
        st.button("Star over", on_click=set_state, args=[0])

    # --------------------------------------------------
    # Screen: Get question for a new ask
    # --------------------------------------------------
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
                    placeholder_choice.empty()
                    set_state(5)

    # --------------------------------------------------
    # Screen: Calls LLM and shows answer for the new ask
    # --------------------------------------------------
    if st.session_state.stage == 5:
        handler = setup_logging()
        st.session_state.ctl_llm.question = st.session_state.question_text
        placeholder_running_again = st.empty()
        with placeholder_running_again.container():
            st.markdown("```Asking with LLM warmed... now is faster.```")
            _, answer = st.session_state.ctl_llm.ask_question_to_llm(
                st.session_state.llm_model_choice
            )
        placeholder_running_again.empty()

        st.write(answer["answer"])
        with st.expander("See details"):
            st.subheader("Details")
            st.write(answer)
            st.divider()
        st.button("Ask again", on_click=set_state, args=[4])
        st.button("Star over", on_click=set_state, args=[0])


if __name__ == "__main__":
    utils = toolkit.UtilsLLM()
    main()
