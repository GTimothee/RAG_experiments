import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def get_db(persist_directory: str):
    emb = OpenAIEmbeddings(
        openai_api_base=os.getenv("embedding_OPENAI_BASE_URL"),
        openai_api_key=os.getenv("embedding_OPENAI_API_KEY"),
        model="gte-large-en-v1.5",
        chunk_size=32,
    )

    if not os.path.isdir(persist_directory):
        raise FileNotFoundError(f"Could not find persist_directory={persist_directory}")
    else:
        print(f"Loading database from disk (at {persist_directory})")
        return Chroma(
            embedding_function=emb,
            persist_directory=persist_directory,
        )
