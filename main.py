import os
import streamlit as st
import pickle
import time
import langchain
from typing import Optional, List
from langchain_core.language_models.llms import LLM
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings 
import google.generativeai as genai
import tempfile

from dotenv import load_dotenv
load_dotenv()

# Configure API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Please set your GOOGLE_API_KEY in the .env file")
    st.stop()

genai.configure(api_key=api_key)

st.title("RefQuest")
st.subheader("Your Research Assistant")

# Source selection
st.sidebar.title("Choose Data Source")
source_option = st.sidebar.selectbox(
    "Select how you want to input data:",
    ["Select an option", "PDF Files", "URLs"]
)

# Initialize session state variables
if 'vectorstore_ready' not in st.session_state:
    st.session_state.vectorstore_ready = False
if 'previous_source_option' not in st.session_state:
    st.session_state.previous_source_option = None
if 'answer_result' not in st.session_state:
    st.session_state.answer_result = None

# Check if source option has changed
if st.session_state.previous_source_option != source_option:
    # Clear the vectorstore state and answer when switching sources
    st.session_state.vectorstore_ready = False
    st.session_state.answer_result = None
    # Remove the vectorstore file if it exists
    if os.path.exists("vectorstore.pkl"):
        os.remove("vectorstore.pkl")
    st.session_state.previous_source_option = source_option

class GeminiLLM(LLM):
    model: str = "gemini-1.5-flash"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            model = genai.GenerativeModel(self.model)
            # Add better instruction to the prompt
            enhanced_prompt = f"""
            You are a helpful assistant that explains concepts clearly and simply.
            When asked about laws, rules, or principles, provide clear explanations in simple terms.
            If the context mentions a law or concept but doesn't explain it fully, use your knowledge to provide a helpful explanation.
            Always provide detailed explanations even if the source material only gives titles or brief mentions.
            
            {prompt}
            
            Instructions: Provide a comprehensive answer. If the context only shows a title or brief statement, 
            expand on it with your knowledge to give a full, helpful explanation.
            """
            response = model.generate_content(enhanced_prompt)
            return response.text
        except Exception as e:
            st.error(f"Error calling Gemini API: {str(e)}")
            return "Error generating response"

    @property
    def _llm_type(self) -> str:
        return "google_gemini_llm"

# Instantiate our custom Gemini LLM
llm = GeminiLLM()

def enhance_question(query, retrieved_docs):
    """Enhance the user's question with context from retrieved documents"""
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    enhanced_query = f"""
    Based on the following context from the document:
    {context}
    
    Original question: {query}
    
    Please provide a clear, simple explanation. If the context only provides a title or brief mention, 
    use your knowledge to explain the concept in detail.
    """
    return enhanced_query

main_placeholder = st.empty()

# Handle PDF input
if source_option == "PDF Files":
    st.sidebar.title("Upload PDF Files")
    uploaded_files = st.sidebar.file_uploader(
        "Choose PDF files", 
        accept_multiple_files=True, 
        type=['pdf']
    )
    
    process_pdfs_clicked = st.sidebar.button("Process PDFs")
    
    if process_pdfs_clicked and uploaded_files:
        file_path = "vectorstore.pkl"
        
        try:
            all_docs = []
            main_placeholder.text("Loading PDF files...")
            
            for uploaded_file in uploaded_files:
                # Save uploaded file to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Load PDF
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()
                all_docs.extend(docs)
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
            
            if not all_docs:
                st.error("No data could be loaded from the uploaded PDFs.")
            else:
                # Split the documents into smaller chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", " ", ""],
                    chunk_size=500,  # Smaller chunks for better context
                    chunk_overlap=50  # Add overlap between chunks
                )
                docs = text_splitter.split_documents(all_docs)
                main_placeholder.text("PDFs loaded and splitting into chunks...")
                
                # Create embeddings and vector store
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                vectorstore = FAISS.from_documents(docs, embeddings)
                
                # Save vectorstore
                with open(file_path, "wb") as f:
                    pickle.dump(vectorstore, f)
                
                st.session_state.vectorstore_ready = True
                main_placeholder.text("Vector store created and saved. You can now ask questions about the PDFs.")
                
        except Exception as e:
            st.error(f"Error processing PDFs: {str(e)}")
    
    elif process_pdfs_clicked and not uploaded_files:
        st.error("Please upload at least one PDF file.")

# Handle URL input
elif source_option == "URLs":
    st.sidebar.title("News Article URLs")
    
    urls = []
    for i in range(3):
        url = st.sidebar.text_input(f"URL {i+1}")
        urls.append(url)
    
    read_urls_clicked = st.sidebar.button("Read Articles")
    
    if read_urls_clicked:
        # Filter out empty URLs
        filtered_urls = [url for url in urls if url.strip()]
        
        if not filtered_urls:
            st.error("Please provide at least one valid URL.")
        else:
            file_path = "vectorstore.pkl"
            
            try:
                # Load the articles from the provided URLs
                loader = UnstructuredURLLoader(urls=filtered_urls)
                main_placeholder.text("Loading articles...")
                
                data = loader.load()
                
                if not data:
                    st.error("No data could be loaded from the provided URLs.")
                else:
                    # Split the documents into smaller chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        separators=["\n\n", "\n", " ", ""],
                        chunk_size=500,  # Smaller chunks for better context
                        chunk_overlap=50  # Add overlap between chunks
                    )
                    docs = text_splitter.split_documents(data)
                    main_placeholder.text("Articles loaded and splitting into chunks...")
                    
                    # Create embeddings and vector store
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                    vectorstore = FAISS.from_documents(docs, embeddings)
                    
                    # Save vectorstore
                    with open(file_path, "wb") as f:
                        pickle.dump(vectorstore, f)
                    
                    st.session_state.vectorstore_ready = True
                    main_placeholder.text("Vector store created and saved. You can now ask questions about the articles.")
                    
            except Exception as e:
                st.error(f"Error processing articles: {str(e)}")

# Question input section - only show if vectorstore is ready OR if file exists
file_path = "vectorstore.pkl"
if st.session_state.vectorstore_ready or os.path.exists(file_path):
    st.subheader("Ask a Question")
    
    # Use session state to control the question input
    query = st.text_input("Question:", key="question_input")
    
    if query:
        try:
            # Load vectorstore
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
            
            # Get relevant documents first
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}  # Retrieve more relevant chunks
            )
            retrieved_docs = retriever.get_relevant_documents(query)
            
            # Enhance the question with context
            enhanced_query = enhance_question(query, retrieved_docs)
            
            # Create chain (removed custom prompt to avoid template error)
            chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True
            )
            
            # Show spinner while processing
            with st.spinner("Generating answer..."):
                # Use the enhanced query instead of the original
                result = chain({"question": enhanced_query}, return_only_outputs=True)
            
            # Store result in session state
            st.session_state.answer_result = result
            
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
            st.error("Please make sure to load data first by processing PDFs or URLs.")
            st.session_state.answer_result = None
    
    # Display results if available
    if st.session_state.answer_result:
        result = st.session_state.answer_result
        
        # Display results
        st.header("Answer")
        st.write(result["answer"])
        
        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                if source.strip():  # Only display non-empty sources
                    st.write(source)
        
        # Optionally display source documents
        if "source_documents" in result:
            with st.expander("View Source Documents"):
                for i, doc in enumerate(result["source_documents"]):
                    st.write(f"**Source {i+1}:**")
                    st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                    if hasattr(doc, 'metadata') and doc.metadata:
                        st.write(f"**Metadata:** {doc.metadata}")
                    st.write("---")
                    
else:
    if source_option == "Select an option":
        st.info("Please select a data source from the sidebar (PDF Files or URLs).")
    elif source_option == "PDF Files":
        st.info("Please upload PDF files and click 'Process PDFs' to get started.")
    elif source_option == "URLs":
        st.info("Please enter URLs and click 'Read Articles' to get started.")