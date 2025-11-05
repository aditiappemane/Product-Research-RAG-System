from typing import Any, Dict,Optional,List
from langchain.chains import RetrievalQA
from langchain.llms import GoogleGemini
from langchain.vectorstores.base import VectorStore
from langchain.embeddings import GoogleGeminiEmbeddings
from langchain.prompts import PromptTemplate
import genai
import os
import langchain.docstore.document
from RAG.app.storage import load_vector_store


llm = GoogleGemini(model=os.getenv("Gemini_Model"))
embeddings = GoogleGeminiEmbeddings()
genai.config.api_key = os.getenv("Google_API_Key")
genai.config.model = os.getenv("Gemini_Model")

def create_qa_chain(vector_store: VectorStore) -> RetrievalQA:
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    prompt_template = """You are an AI assistant that helps users find information from their documents.
    Use the following context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa_chain

def matches_metadata(doc: langchain.docstore.document.Document, metadata: Dict[str, Any]) -> bool:
    for key, value in metadata.items():
        if doc.metadata.get(key) != value:
            return False
    return True
def filter_documents_by_metadata(documents: List[langchain.docstore.document.Document], metadata: Dict[str, Any]) -> List[langchain.docstore.document.Document]:
    filtered_docs = [doc for doc in documents if matches_metadata(doc, metadata)]
    return filtered_docs

RAG_Prompt = """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}   
Question: {question}
Answer:"""
def generate_answer(vector_store: VectorStore, question: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    qa_chain = create_qa_chain(vector_store)
    if metadata:
        all_docs = vector_store.similarity_search(question, k=10)
        filtered_docs = filter_documents_by_metadata(all_docs, metadata)
        if not filtered_docs:
            return {"answer": "I don't know.", "source_documents": []}
        context = "\n\n".join([doc.page_content for doc in filtered_docs])
        prompt = RAG_Prompt.format(context=context, question=question)
        answer = llm.predict(prompt)
        return {"answer": answer, "source_documents": filtered_docs}
    else:
        result = qa_chain.run(question)
        return {"answer": result['result'], "source_documents": result['source_documents']}