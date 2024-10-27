import os
import pandas as pd
from uuid import uuid4

from tqdm import tqdm
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

LIMIT_DB_SIZE = 1000


def make_db(emb, persist_directory):
    print("Preparing chunks...")
    documents = DatasetMaker.make()
    print(f"Total nb chunks: {len(documents)}")

    if LIMIT_DB_SIZE:
        print(f"LIMIT_DB_SIZE={LIMIT_DB_SIZE}. Truncating.")
        documents = documents[:LIMIT_DB_SIZE]

    print(f"Building DB at {persist_directory}...")
    chroma_db = Chroma(
        collection_name="langchain",
        embedding_function=emb,
        persist_directory=persist_directory,
    )
    for i in range(0, len(documents), 100):  # Loading in chunks to avoid OOM error
        indices = i, i + 100
        print(f"\t- Ingesting docs {indices[0]} to {indices[1]}...")
        batch = documents[indices[0] : indices[1]]
        if not len(batch):
            break
        uuids = [str(uuid4()) for _ in range(len(batch))]
        chroma_db.add_documents(batch, ids=uuids)
    print("Done.")
    return chroma_db


def get_db(persist_directory: str):
    emb = OpenAIEmbeddings(
        openai_api_base=os.getenv("embedding_OPENAI_BASE_URL"),
        openai_api_key=os.getenv("embedding_OPENAI_API_KEY"),
        model="gte-large-en-v1.5",
        chunk_size=32,
    )

    if not os.path.isdir(persist_directory):
        return make_db(emb, persist_directory)
    else:
        print(f"Loading database from disk (at {persist_directory})")
        return Chroma(
            embedding_function=emb,
            persist_directory=persist_directory,
        )


class DatasetMaker:
    """From https://huggingface.co/learn/cookbook/en/advanced_rag"""

    MARKDOWN_SEPARATORS = [
        "\n#{1,6} ",
        "```\n",
        "\n\\*\\*\\*+\n",
        "\n---+\n",
        "\n___+\n",
        "\n\n",
        "\n",
        " ",
        "",
    ]

    def make(chunk_size=512):

        print("\t- Loading data from file...")
        dataset = datasets.load_dataset("csv", data_files="huggingface_doc.csv")[
            "train"
        ]
        dataset = [
            Document(page_content=doc["text"], metadata={"source": doc["source"]})
            for doc in tqdm(dataset)
        ]
        print(f"\t- Dataset size: {len(dataset)}")

        print("\t- Splitting...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size / 10),
            add_start_index=True,
            strip_whitespace=True,
            separators=DatasetMaker.MARKDOWN_SEPARATORS,
        )
        documents = text_splitter.split_documents(dataset)

        print("\t- Dropping duplicates...")
        unique_texts = {}
        uniq = []
        for doc in documents:
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                uniq.append(doc)

        return uniq


if __name__ == "__main__":
    load_dotenv()
    persist_directory = (
        f"./chroma_db_{LIMIT_DB_SIZE}" if LIMIT_DB_SIZE is not None else "./chroma_db"
    )
    chroma = get_db(persist_directory=persist_directory)

    # test
    docs = chroma.similarity_search(
        query="what do you know about huggingface endpoint?", k=5
    )
    print(docs)
