import os 

from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from build_vector_database import get_db

"""Sources:
- https://huggingface.co/learn/cookbook/en/advanced_rag
- https://python.langchain.com/docs/tutorials/rag/#retrieval-and-generation-generate
"""


SYSTEM_PROMPT = """Using the information contained in the context,

give a comprehensive answer to the question.

Respond only to the question asked, response should be concise and relevant to the question.

Provide the number of the source document when relevant.

If the answer cannot be deduced from the context, do not give an answer."""

USER_PROMPT = """Context:

{context}

---

Now here is the question you need to answer.

Question: {question}
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    load_dotenv()
    db = get_db("chroma_db_1000")
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatOpenAI(
        openai_api_base=os.getenv("OPENAI_BASE_URL"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name="Llama-3-70B-Instruct", 
        temperature=0.0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ]
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )   
    
    output = rag_chain.invoke("What do you know about huggingface endpoints?")
    print(output)
    