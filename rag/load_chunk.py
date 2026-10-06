import os
import pickle

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

import config
from store_logs.db import collection
from rag.cache_file import get_cached_embeddings

def load_and_chunk(pdf_path:str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            collection.insert_one({
                "file" : "not found",
                "file_name" : pdf_path
            })
            
        )

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    #logs
    collection.insert_one({
        "loaded_pdf_path" : pdf_path,
        "length_of_pages" : len(pages)
    })

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = config.CHUNK_SIZE,
        chunk_overlap = config.CHUNK_OVERLAP,
        seperator = ["\n\n","\n","."," ",""]
    )

    chunks = splitter.split_documents(pages)

    #logs
    collection.insert_one({
        "no_of_chunks" : len(chunks),
        "chunk_size" : config.CHUNK_SIZE,
        "chunk_overlap" : config.CHUNK_OVERLAP
    })


def build_index(force_rebuild : bool = False):
    os.mkdirs(config.STORE_DIR,is_exists = True)

    if (not force_rebuild
        and os.path.exists(config.FAISS_DIR)
        and os.path.exists(config.BM25_DOCS_PATH)):
        collection.insert_one({
            "found_file" : "yes and skip rebuilding"
        })
        return
    
    chunks = load_and_chunk(config.PDF_PATH)
    embeddings = get_cached_embeddings()

    vectorstore = FAISS.from_documents(chunks,embeddings)
    vectorstore.save_local(config.FAISS_DIR)

    collection.insert_one({
        "semantic_store" : config.FAISS_DIR
    })

    with open(config.BM25_DOCS_PATH,"wb") as f:
        pickle.dump(chunks,f)

    collection.insert_one({
        "bm25_chunks" : config.BM25_DOCS_PATH
    })

if __name__ == "__main__":
    build_index(force_rebuild = True)