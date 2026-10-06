import os
import pickle

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

import config
from store_logs.db import collection

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