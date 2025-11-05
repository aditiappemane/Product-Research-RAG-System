import os
from langchain. text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import GoogleGeminiEmbeddings
from langchain.docstore.document import Document

genai.config.api_key = os.getenv("Google_API_Key")
genai.config.model = os.getenv("Gemini_Model")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
)

def generate_embeddings(docs, persist_directory):
    texts = text_splitter.split_documents(docs)
    embeddings = GoogleGeminiEmbeddings()
    vector_store = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    vector_store.persist()
    return vector_store

def load_vector_store(persist_directory):
    embeddings = GoogleGeminiEmbeddings()
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
    return vector_store
def create_document_from_text(text, metadata=None):
    if metadata is None:
        metadata = {}
    document = Document(page_content=text, metadata=metadata)
    return document
    
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk, metadata=metadata) for chunk in chunks]
    
    for i, doc in enumerate(documents):
        doc.metadata['chunk_index'] = i 
        metadata={
            'source': 'user_upload',
            'original_text': text
            "price":Price        }
    return documents
