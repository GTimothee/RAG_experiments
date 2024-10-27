from dotenv import load_dotenv
from library.rag import get_rag_chain


if __name__ == "__main__":
    load_dotenv()
    rag_chain = get_rag_chain(chroma_db_dirpath="data/chroma_db_1000")
    output = rag_chain.invoke("What do you know about huggingface endpoints?")
    print(output)
