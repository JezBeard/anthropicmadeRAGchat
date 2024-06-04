import streamlit as st
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Set up OpenAI API key
openai_api_key = "YOUR_OPENAI_API_KEY"

# Set up Streamlit app
st.title("Document Q&A with OpenAI")

# File upload
uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])

if uploaded_file is not None:
    # Load and process the document
    loader = UnstructuredFileLoader(uploaded_file)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Create embeddings and store them in FAISS
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(texts, embeddings)

    # Set up the question-answering chain
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=openai_api_key),
        chain_type="stuff",
        retriever=db.as_retriever(),
    )

    # User question input
    question = st.text_input("Ask a question about the document:")

    if question:
        # Generate answer
        answer = qa.run(question)
        st.write("Answer:", answer)
