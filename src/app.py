import streamlit as st
import controller as controller_llm
import utils as toolkit
import parameters as general_parameters
import pdfplumber
import logging
from custom_log_handler import StreamlitLogHandler
from urllib.parse import urlparse
import time

# Set page config at the top level (only once)
if 'page_config_set' not in st.session_state:
    st.set_page_config(
        page_title="LLM Question Answering App",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.session_state.page_config_set = True

logging.basicConfig(level=logging.ERROR)

def is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def setup_logging(log_container):
    root_logger = logging.getLogger()  # Get the root logger
    handler = StreamlitLogHandler(log_container)
    handler.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    return handler

def main():
    # Add custom CSS
    st.markdown("""
        <style>
        .stProgress > div > div > div > div {
            background-color: #4CAF50;
        }
        .tooltip {
            position: relative;
            display: inline-block;
            border-bottom: 1px dotted black;
        }
        </style>
    """, unsafe_allow_html=True)

    def set_state(i):
        st.session_state.stage = i

    def set_content_type_choice_visibility(par_value):
        st.session_state["show_content_type_choice"] = par_value

    def set_state_and_content_type_choice_visibility(
        par_state_value, par_value_show_content_type_choice
    ):
        set_state(par_state_value)
        set_content_type_choice_visibility(par_value_show_content_type_choice)

    def set_state_and_content_type_choice_visibility_and_rerun(
        par_state_value, par_value_show_content_type_choice
    ):
        set_state(par_state_value)
        set_content_type_choice_visibility(par_value_show_content_type_choice)
        st.rerun()

    def set_llm_model_choice():
        st.session_state.llm_model_choice = st.session_state["element_llm_model_choice"]

    def set_source_type_choice():
        st.session_state.source_type_choice = st.session_state[
            "element_source_type_choice"
        ]

    def set_text_handling_choice():
        st.session_state.text_handling_choice = st.session_state[
            "element_text_handling_choice"
        ]

    def reset_controller():
        """ Reset controller to avoid past questions interfering in the next question """
        st.session_state.ctl_llm.chain_is_prepared = False

    # Initialize session state
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
    if "show_content_type_choice" not in st.session_state:
        st.session_state["show_content_type_choice"] = True

    # Main UI
    st.title("🤖 LLM Question Answering App")
    
    # Progress bar
    progress_text = ["Input Source", "Question", "Confirm", "Processing", "Results"]
    progress = st.progress(0)
    progress.progress(min(st.session_state.stage / 4, 1.0))
    st.caption(f"Step {st.session_state.stage + 1} of 5: {progress_text[min(st.session_state.stage, 4)]}")

    # Sidebar with settings
    with st.sidebar:
        st.header("Settings")
        placeholder_choice = st.empty()
        with placeholder_choice.container():
            if st.session_state.stage not in general_parameters.par__stages_when_choices_are_disabled:
                options_model_choice = ["Gemini", "chatGPT"]
                default_index = options_model_choice.index(st.session_state.llm_model_choice) if st.session_state.llm_model_choice in options_model_choice else 0
                st.radio(
                    "LLM Model",
                    options_model_choice,
                    on_change=set_llm_model_choice,
                    key="element_llm_model_choice",
                    index=default_index,
                    help="Choose the language model to use for answering questions"
                )
                
                if st.session_state["show_content_type_choice"]:
                    options_content_type = ["URL", "PDF"]
                    default_index = options_content_type.index(st.session_state.source_type_choice) if st.session_state.source_type_choice in options_content_type else 0
                    st.radio(
                        "Source Type",
                        options_content_type,
                        on_change=set_source_type_choice,
                        key="element_source_type_choice",
                        index=default_index,
                        help="Choose the type of content to analyze"
                    )
                
                options_content_handling = [
                    general_parameters.par__label_content_handling_retrieved_documents,
                    general_parameters.par__label_content_handling_all_text,
                ]
                default_index = options_content_handling.index(st.session_state.text_handling_choice) if st.session_state.text_handling_choice in options_content_handling else 0
                st.radio(
                    "Content Handling",
                    options_content_handling,
                    on_change=set_text_handling_choice,
                    key="element_text_handling_choice",
                    index=default_index,
                    help="Choose how to process the content"
                )

    # Main content area
    placeholder_input_content = st.empty()
    placeholder_input_question = st.empty()
    placeholder_input_confirm = st.empty()
    placeholder_log = st.empty()
    placeholder_calling_llm = st.empty()
    placeholder_input_new_question = st.empty()
    placeholder_running_again = st.empty()

    # --------------------------------------------------
    # Screen: Get content source
    # --------------------------------------------------
    if st.session_state.stage < 1:
        reset_controller()
        st.session_state["show_content_type_choice"] = True
        with placeholder_input_content.container():
            if st.session_state.source_type_choice == "URL":
                st.subheader("Article URL")
                st.write("Enter the URL of the article you want to ask about:")
                with st.form("form_content_url", border=False, clear_on_submit=True):
                    url_to_ask = st.text_input("URL:", help="Enter a valid URL starting with http:// or https://")
                    submitted = st.form_submit_button("Submit")
                    if submitted:
                        if not url_to_ask:
                            url_to_ask = general_parameters.par__default_url_content_to_test
                        if not is_valid_url(url_to_ask):
                            st.error("Please enter a valid URL starting with http:// or https://")
                        else:
                            st.session_state.content_ref_to_ask = url_to_ask
                            placeholder_input_content.empty()
                            set_state_and_content_type_choice_visibility_and_rerun(1, False)
            
            elif st.session_state.source_type_choice == "PDF":
                st.subheader("PDF File")
                st.write("Upload a PDF file to ask questions about:")
                file_ref = st.file_uploader("Upload PDF", type=["pdf"], help="Upload a PDF file (max 10MB)")
                if st.button("Process"):
                    if file_ref is not None:
                        if file_ref.size > 10 * 1024 * 1024:  # 10MB limit
                            st.error("File size too large. Please upload a file smaller than 10MB.")
                        else:
                            with st.spinner("Processing PDF..."):
                                st.session_state.content_ref_to_ask = file_ref.name
                                st.session_state.pdf_file_ref = file_ref
                                file_details = {
                                    "Filename": file_ref.name,
                                    "File Type": file_ref.type,
                                    "File Size": f"{file_ref.size / 1024:.1f} KB"
                                }
                                st.json(file_details)
                                try:
                                    with pdfplumber.open(file_ref) as f:
                                        pages = f.pages[0]
                                        st.subheader("Preview (Page 1):")
                                        st.text(pages.extract_text())
                                    set_state_and_content_type_choice_visibility_and_rerun(1, False)
                                except Exception as e:
                                    st.error(f"Error processing PDF: {str(e)}")
                    else:
                        st.warning("Please upload a PDF file first.")

    # --------------------------------------------------
    # Screen: Get question to be asked
    # --------------------------------------------------
    if st.session_state.stage == 1:
        with placeholder_input_question.container():
            st.subheader("Question")
            st.write("What would you like to know about the content?")
            with st.form("form_question_first", border=False, clear_on_submit=True):
                text_question = st.text_input("Question:", help="Enter your question about the content")
                submitted = st.form_submit_button("Submit")
                if submitted:
                    if not text_question.strip():
                        st.error("Please enter a question")
                    else:
                        st.session_state.question_text = text_question
                        set_state_and_content_type_choice_visibility_and_rerun(2, False)
            st.button(
                "Start Over",
                on_click=set_state_and_content_type_choice_visibility,
                args=[0, True],
            )

    # --------------------------------------------------
    # Screen: Confirm source and question
    # --------------------------------------------------
    if st.session_state.stage == 2:
        placeholder_input_question.empty()
        with placeholder_input_confirm.container():
            st.subheader("Confirm")
            st.markdown(
                f"**Question:** {st.session_state.question_text}\n\n"
                f"**Source:** {st.session_state.content_ref_to_ask}"
            )
            col1, col2 = st.columns(2)
            with col1:
                st.button("Ask to LLM", on_click=set_state, args=[3])
            with col2:
                st.button("Start Over", on_click=set_state_and_content_type_choice_visibility, args=[0, True])

    # --------------------------------------------------
    # Screen: Prepare content, call LLM and show answer
    # --------------------------------------------------
    if st.session_state.stage == 3:
        setup_logging(placeholder_log)
        placeholder_choice.empty()
        with placeholder_calling_llm.container():
            with st.spinner("Processing your question..."):
                result_success, answer = st.session_state.ctl_llm.ask_to_llm(
                    st.session_state.content_ref_to_ask,
                    st.session_state.question_text,
                    st.session_state.llm_model_choice,
                    st.session_state.source_type_choice,
                    st.session_state.pdf_file_ref,
                    st.session_state.text_handling_choice,
                )

                if result_success == -1:
                    st.error("This article is not accessible. Please try another article.")
                    time.sleep(general_parameters.par__waiting_time_in_seconds_in_error_case)
                    set_state(0)
                    st.experimental_rerun()
                elif result_success == -2:
                    st.error("Vector store connection error. Please check the connection.")
                    time.sleep(general_parameters.par__waiting_time_in_seconds_in_error_case)
                    set_state(0)
                    st.experimental_rerun()
                elif result_success == -3:
                    st.error("Unknown content type error.")
                    raise Exception("Unknown content type to get content from.")

        # Present answer
        st.subheader("Answer")
        if st.session_state.text_handling_choice == general_parameters.par__label_content_handling_all_text:
            st.write(answer)
        elif st.session_state.text_handling_choice == general_parameters.par__label_content_handling_retrieved_documents:
            st.write(answer["answer"])
            with st.expander("View Details"):
                st.json(answer)
        
        col1, col2 = st.columns(2)
        with col1:
            st.button("Ask Another Question", on_click=set_state_and_content_type_choice_visibility, args=[4, False])
        with col2:
            st.button("Start Over", on_click=set_state_and_content_type_choice_visibility, args=[0, True])

    # --------------------------------------------------
    # Screen: Get question for a new ask
    # --------------------------------------------------
    if st.session_state.stage == 4:
        placeholder_log.empty()
        placeholder_calling_llm.empty()
        with placeholder_input_new_question.container():
            st.subheader("Question")
            st.write("What more do you want to know about the article?")
            with st.form("form_question_again", border=False, clear_on_submit=True):
                text_question = st.text_input("Question:")
                submitted = st.form_submit_button("Submit")
                if submitted:
                    st.session_state.question_text = text_question
                    placeholder_choice.empty()
                    set_state(5)

    # --------------------------------------------------
    # Screen: Calls LLM and shows answer for the new ask
    # --------------------------------------------------
    if st.session_state.stage == 5:
        placeholder_input_new_question.empty()
        setup_logging(placeholder_log)
        st.session_state.ctl_llm.question = st.session_state.question_text
        st.session_state.ctl_llm.content_handling_choice = (
            st.session_state.text_handling_choice
        )
        with placeholder_running_again.container():
            if (
                st.session_state.text_handling_choice
                == general_parameters.par__label_content_handling_retrieved_documents
            ):
                (
                    _,
                    answer,
                ) = st.session_state.ctl_llm.ask_question_to_llm_using_vector_store(
                    st.session_state.llm_model_choice,
                    st.session_state.source_type_choice,
                )
            elif (
                st.session_state.text_handling_choice
                == general_parameters.par__label_content_handling_all_text
            ):
                (
                    _,
                    answer,
                ) = st.session_state.ctl_llm.ask_question_to_llm_passing_limited_text(
                    st.session_state.llm_model_choice,
                    st.session_state.source_type_choice,
                )
            else:
                raise Exception("Unknown text handling type.")
        placeholder_running_again.empty()

        # Present answer
        if (
            st.session_state.text_handling_choice
            == general_parameters.par__label_content_handling_all_text
        ):
            st.write(answer)
        elif (
            st.session_state.text_handling_choice
            == general_parameters.par__label_content_handling_retrieved_documents
        ):
            st.write(answer["answer"])
            with st.expander("See details"):
                st.subheader("Details")
                st.write(answer)
                st.divider()
        st.button("Ask again", on_click=set_state, args=[4])
        st.button(
            "Start over",
            on_click=set_state_and_content_type_choice_visibility,
            args=[0, True],
        )


if __name__ == "__main__":
    utils = toolkit.UtilsLLM()
    main()
