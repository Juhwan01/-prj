# streamlit_app.py

import streamlit as st
import tempfile
import os
from pdf_processor import process_pdf, create_qa_word_document

def main():
    st.title("PDF Q&A Extractor")

    # 슬라이더의 최대값을 9로 조정합니다 (실제로는 1부터 10까지 선택 가능)
    num_questions = st.slider("Number of questions to generate", min_value=1, max_value=10, value=5, step=1)

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        if st.button("Process PDF"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            with st.spinner("Processing PDF..."):
                result = process_pdf(tmp_file_path, num_questions)

            os.unlink(tmp_file_path)  # Clean up the temporary file

            if "qa_pairs" in result:
                st.subheader(f"Extracted {len(result['qa_pairs'])} Q&A Pairs:")
                for pair in result['qa_pairs']:
                    st.write(f"Q: {pair['question']}")
                    st.write(f"A: {pair['answer']}")
                    st.write("---")

                create_qa_word_document(result)

                with open("qa_output.docx", "rb") as file:
                    st.download_button(
                        label="Download Q&A as Word Document",
                        data=file,
                        file_name="qa_pairs.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
            else:
                st.error("No Q&A pairs were extracted from the document.")

if __name__ == "__main__":
    main()