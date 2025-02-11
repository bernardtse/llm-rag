import streamlit as st
import os
import atexit
import shutil
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
import pdfplumber
from docx import Document
from odf.opendocument import load as odt_load
from odf.text import P

# Configure AI Models
LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"

# Paths for storage
CHROMA_DB_PATH = "./chroma_db"
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# Define cleanup function
def cleanup_chroma_db():
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
        print("ChromaDB folder deleted on app exit.")

# Register cleanup function at exit
atexit.register(cleanup_chroma_db)

# Initialise the embedding model
embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)

# Function to initialise and return the vector store
def get_vector_store():
    try:
        return Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_model)
    except Exception as e:
        st.toast(f"Error initialising vector store: {e}")
        return None

# Initialise vector store
vector_store = get_vector_store()

# Function to process and store text-based data
def process_data(source, content, metadata=None):
    try:
        # Check if the content is empty or just whitespace
        if not content.strip():  # .strip() removes any leading/trailing whitespace
            st.warning(f"{source} file is empty or contains no text.")
            return  # Return early if content is empty
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.create_documents([content])
        
        if metadata:
            for chunk in chunks:
                chunk.metadata = metadata
        
        vector_store.add_documents(chunks)
        st.toast(f"{source} content processed and stored in ChromaDB!")
    except Exception as e:
        st.toast(f"Error processing {source}: {e}")

# Function to handle TXT and MD file uploads
def handle_text_upload(uploaded_file):
    content = uploaded_file.getvalue().decode("utf-8")
    
    if not content.strip():  # Check if the content is empty
        st.warning("The uploaded text file is empty.")
        return  # Return early if content is empty
    
    process_data("Text File", content)
    
# Function to handle PDF uploads
def handle_pdf_upload(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        content = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    if not content.strip():  # Check if PDF content is empty
        st.warning("The uploaded PDF is empty or contains no text.")
        return  # Return early if content is empty    
    process_data("PDF", content)

# Function to handle DOCX file uploads
def handle_docx_upload(uploaded_file):
    doc = Document(uploaded_file)
    content = "\n".join([para.text for para in doc.paragraphs])
    
    if not content.strip():  # Check if the content is empty
        st.warning("The uploaded DOCX file is empty.")
        return  # Return early if content is empty
    
    process_data("DOCX", content)

# Function to handle ODT file uploads
def handle_odt_upload(uploaded_file):
    try:
        odt_doc = odt_load(uploaded_file)
        content = ""
        # Iterate over all paragraph elements in the ODT file
        for elem in odt_doc.getElementsByType(P):
            # Get text content from the paragraph element
            if elem.firstChild is not None:
                # Check if the first child is a text node (and handle it)
                if elem.firstChild.nodeType == 3:  # TEXT_NODE (Node type 3 represents text nodes)
                    content += elem.firstChild.data + "\n"
        if not content.strip():  # Check for empty content
            st.warning("The uploaded ODT file is empty.")
            return
        process_data("ODT", content)
    except Exception as e:
        st.toast(f"Error processing ODT file: {e}")

# Function to handle RTF uploads
def handle_rtf_upload(uploaded_file):
    content = uploaded_file.getvalue().decode("utf-8")
    
    if not content.strip():  # Check if the content is empty
        st.warning("The uploaded RTF file is empty.")
        return  # Return early if content is empty
    
    process_data("RTF", content)

# Function to scrape and process webpages
def scrape_webpage(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # to raise HTTPError for bad responses
    except requests.exceptions.RequestException as e:
        st.toast(f"Error during webpage request: {e}")
        return

    try:
        soup = BeautifulSoup(response.content, "html.parser")
        content = soup.get_text(separator="\n", strip=True)
        
        if not content.strip():  # Check if the content is empty
            st.warning("The scraped webpage is empty.")
            return  # Return early if content is empty
    except Exception as e:
        st.toast(f"Error during BeautifulSoup parsing: {e}")
        return

    # Process the non-empty content
    process_data("Webpage", content, metadata={"source": url})

# Function to query the vector store and display results
def query_data(query):
    if vector_store is None:
        st.warning("Vector store not initialised.")
        return
    
    try:
        results = vector_store.similarity_search(query, k=5)
        if not results:
            st.warning("No results found in ChromaDB.")
            return
        
        # Generate response using Ollama
        llm = OllamaLLM(model=LLM_MODEL)
        context = "\n\n".join([f"**Result {i+1}:** {r.page_content}" for i, r in enumerate(results)])
        response = llm.invoke(f"Based on the following context, answer the question: {query}\n\nContext:\n{context}")
        
        st.subheader("Ollama Response:")
        st.write(response)

        # Add a toggle to show context (query results)
        with st.expander("Show context (query results)"):
            st.write(context)
            
    except Exception as e:
        st.toast(f"Error during querying: {e}")

# Function to reset the vector store
def reset_vector_store():
    try:
        # Clear Chroma database
        global vector_store
        if vector_store is not None:
            vector_store.delete_collection()
        
        # Re-initialise ChromaDB client and vector store
        vector_store = get_vector_store()

        st.toast("ChromaDB has been cleared and reinitialised!")

    except Exception as e:
        st.toast(f"Error resetting vector store: {e}")

# Function to reset session state
def reset():
    # Increment reset key
    st.session_state["reset"] += 1

    # Force a page reload after resetting
    st.rerun()

# Force run reset_vector_store on every page load
reset_vector_store()

# Interface for text-based files
def text_file_interface():
    st.header("Upload and Process Text Files")
    uploaded_file = st.file_uploader("Upload a Text File (.txt / .md)", type=["txt", "md"], key=f"text_file_upload_{st.session_state.reset}")
    if uploaded_file:
        handle_text_upload(uploaded_file)
    query = st.text_input("Query Text Files", key=f"text_query_{st.session_state.reset}")
    if query:
        query_data(query)
    if st.button("Reset") and 'reset_flag' not in st.session_state:
        reset()
        st.session_state['reset_flag'] = True

# Interface for document-based files
def document_file_interface():
    st.header("Upload and Process Document Files")
    uploaded_file = st.file_uploader("Upload a Document (.pdf / .docx / .odt / .rtf)", type=["pdf", "docx", "odt", "rtf"], key=f"document_file_upload_{st.session_state.reset}")
    if uploaded_file:
        # Check the file type first by MIME type
        if uploaded_file.type == "application/pdf" or uploaded_file.name.endswith('.pdf'):
            handle_pdf_upload(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or uploaded_file.name.endswith('.docx'):
            handle_docx_upload(uploaded_file)
        elif uploaded_file.type == "application/vnd.oasis.opendocument.text" or uploaded_file.name.endswith('.odt'):
            handle_odt_upload(uploaded_file)
        elif uploaded_file.type == "text/rtf" or uploaded_file.name.endswith('.rtf'):
            handle_rtf_upload(uploaded_file)
        else:
            st.warning("Unsupported file type. Please upload a PDF, DOCX, ODT, or RTF file.")
    query = st.text_input("Query Documents", key=f"document_query_{st.session_state.reset}")
    if query:
        query_data(query)
    if st.button("Reset") and 'reset_flag' not in st.session_state:
        reset()
        st.session_state['reset_flag'] = True

# Interface for web scraping
def web_scraping_interface():
    st.header("Scrape and Process Webpages")
    url = st.text_input("Enter a webpage URL (full URL)", key=f"url_{st.session_state.reset}")
    if url:
        scrape_webpage(url)
    query = st.text_input("Query Webpages", key=f"url_query_{st.session_state.reset}")
    if query:
        query_data(query)
    if st.button("Reset") and 'reset_flag' not in st.session_state:
        reset()
        st.session_state['reset_flag'] = True

# Main UI
st.title("AI-Powered Multi-Mode RAG App")

# Initialise session state for reset on page load
if "reset" not in st.session_state:
    st.session_state["reset"] = 0

# Switch to the correct interface based on selected mode
mode = st.sidebar.radio("Choose Mode", ["Text Files", "Documents", "Web Scraping"], key="mode")

if mode == "Text Files":
    text_file_interface()
elif mode == "Documents":
    document_file_interface()
elif mode == "Web Scraping":
    web_scraping_interface()