from dotenv import load_dotenv
from library.keyword_retriever import KeywordRetriever


if __name__ == "__main__":
    load_dotenv()
    retriever = KeywordRetriever('data/index_1000')
    docs = retriever.invoke("What do you know about huggingface endpoints?")
    print(docs)