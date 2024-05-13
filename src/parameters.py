""" GeneralParameters"""

par__vector_store_index_name = "llm-101-poc"
par__pdf_file_path = "../data/articleAccessibleDesign.pdf"
par__verbose_mode = True
par__prompt_template = """Answer the question as precise as possible using the provided context. If the answer is
                    not contained in the context, say "Answer not available in context" \n\n
                    Context: \n {context}?\n
                    Question: \n {input} \n
                    Answer:
                  """
par__prompt_template_var_context = "context"
par__prompt_template_var_input = "input"
par__verbose_mode = True
par__waiting_time_in_seconds = 5
par__waiting_time_in_seconds_in_error_case = 10
par__chunk_size = 1000
par__chunk_overlap = 200
