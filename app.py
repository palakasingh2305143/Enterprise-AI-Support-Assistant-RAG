import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import os

 

OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

 

st.set_page_config(page_title="Enterprise AI Support Assistant")

st.title("Enterprise AI Support Assistant")
st.write("Upload company documents and ask questions.")

uploaded_file = st.file_uploader(
    "Upload PDF Knowledge Base",
    type=["pdf"]
)

 

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    retriever = vector_db.as_retriever(
        search_kwargs={"k": 4}
    )

    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    st.success("Knowledge Base Loaded Successfully")

    query = st.text_input(
        "Ask a Question"
    )

    if query:

        with st.spinner("Generating Answer..."):

            answer = qa_chain.run(query)

        st.subheader("Response")

        st.write(answer)

        with st.expander("Retrieved Chunks"):

            docs = retriever.get_relevant_documents(query)

            for i, doc in enumerate(docs):

                st.write(f"### Chunk {i+1}")

                st.write(doc.page_content[:1000])

 

st.sidebar.title("Project Features")

st.sidebar.markdown("""
✅ Conversational AI

✅ Retrieval-Augmented Generation (RAG)

✅ ChromaDB Vector Database

✅ Semantic Search

✅ Enterprise Knowledge Assistant

✅ PDF Question Answering

✅ LangChain Pipeline

✅ OpenAI GPT Integration
""")
