import streamlit as st
import pdfplumber

def main():
    st.title("LLM very simple app")
    st.subheader("PDF file")
    file_ref = st.file_uploader("Upload PDF", type=["pdf"])
    if st.button("Process"):
        if file_ref is not None:
            # Print pdf file metadata
            file_details = {
            "filename": file_ref.name
            ,"filetype": file_ref.type
            ,"filesize": file_ref.size
            }
            st.write(file_details)

            # Load pdf file
            try:
                with pdfplumber.open(file_ref) as f:
                    pages = f.pages[0]
                    st.subheader("Page 1:")
                    st.markdown("""---""")
                    st.write(pages.extract_text())
            except Exception as e:
                st.write(">>Error loading pdf file.")
                raise e
        else:
            st.write("Please, upload a PDF file first.")




if __name__ == '__main__':
    main()