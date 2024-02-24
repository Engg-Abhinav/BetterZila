import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.cohere import Cohere
import PyPDF2

@st.cache(allow_output_mutation=True)
def get_pdf_text(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

@st.cache(allow_output_mutation=True)
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=20,
        length_function=len
    )
    return text_splitter.split_text(text)

@st.cache(allow_output_mutation=True)
def get_vectorstore(text_chunks):
    embeddings = CohereEmbeddings()
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def main():
    st.title("Chat with PDF :books:")
    uploaded_file = st.file_uploader("Choose your PDF file", type="pdf")
    user_question = st.text_input("Ask a question about your document:")

    if st.button("Process") and uploaded_file and user_question:
        with st.spinner("Extracting text from PDF..."):
            raw_text = get_pdf_text(uploaded_file)
        with st.spinner("Processing text chunks..."):
            text_chunks = get_text_chunks(raw_text)
        with st.spinner("Generating vector store..."):
            vectorstore = get_vectorstore(text_chunks)

        llm = Cohere(cohere_api_key=st.secrets["COHERE_API_KEY"])
        conversation_chain = RetrievalQA.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
        )

        prompt = (f"You will be asked a question by the user.\n"
                  "The answer should be 3 - 4 sentences according to the context provided.\n"
                  "The answer should be in your own words.\n"
                  "The answer should be easy to understand.\n"
                  "The answer should be grammatically correct.\n"
                  "The answer should be relevant to the question.\n"
                  f"The question is {user_question}.")
        with st.spinner("Fetching your answer..."):
            answer = conversation_chain.invoke({"query": prompt})
            st.info(answer["result"])

if __name__ == '__main__':
    main()