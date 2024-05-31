""" General Parameters"""

par__vector_store_index_name = "llm-101-poc"
par__pdf_file_path = "../data/articleAccessibleDesign.pdf"
par__verbose_mode = True
par__prompt_template_generic_chain = """Answer the question as precise as possible using the provided context. If the answer is
                    not contained in the context, say "Answer not available in context" \n\n
                    Context: \n {context}?\n
                    Question: \n {input} \n
                    Answer:
                  """
par__prompt_template_var_context = "context"
par__prompt_template_var_input = "input"
par__waiting_time_in_seconds = 5
par__waiting_time_in_seconds_in_error_case = 10
par__chunk_size = 1000
par__chunk_overlap = 200
par__default_llm_model_choice = "Gemini"
par__default_source_type_choice = "URL"
par__default_url_content_to_test = "https://staffeng.com/guides/staff-archetypes/"
par__stages_when_choices_are_disabled = [3, 5]
par__log_textarea_UI_key = "log_textarea"
par__label_content_handling_all_text = "All text"
par__label_content_handling_retrieved_documents = "Filter relevant parts"
par__default_text_handling_choice = par__label_content_handling_retrieved_documents
par__limit_length_text_content = 10000
