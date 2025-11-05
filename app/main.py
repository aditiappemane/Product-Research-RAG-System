import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from RAG.app.storage import generate_embeddings, load_vector_store, create_document_from_text
from RAG.app.rag import generate_answer
from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStore
import httpx

from app.crawler import Crawler 

app = FastAPI()

##Crawler Layer
class CrawlRequest(BaseModel):
    url: str
    depth: Optional[int] = 1


class queryRequest(BaseModel):
    question: str
    metadata: Optional[Dict[str, Any]] = None
    persist_directory: str


@app.post("/crawl")
async def crawl_endpoint(request: CrawlRequest):
    try:
        crawler = Crawler(max_depth=request.depth)
        crawled_text = await crawler.crawl(request.url)
        if not crawled_text:
            raise HTTPException(status_code=404, detail="No text found at the provided URL.")
        document = create_document_from_text(crawled_text, metadata={"source_url": request.url})
        return {"message": "Crawling successful.", "document_metadata": document.metadata}
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"HTTP error occurred: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    
@app.post("/query")
def query_endpoint(request: queryRequest):
    try:
        vector_store: VectorStore = load_vector_store(request.persist_directory)
        result = generate_answer(vector_store, request.question, request.metadata)
        return {"answer": result["answer"], "source_documents": [doc.metadata for doc in result["source_documents"]]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    