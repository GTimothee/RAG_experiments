import os
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import argparse

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

from library.database import get_db

"""Sources:
- https://huggingface.co/learn/cookbook/en/advanced_rag
- https://python.langchain.com/docs/tutorials/rag/#retrieval-and-generation-generate
"""

parser = argparse.ArgumentParser()
parser.add_argument("chroma_dir", help="path to chroma db")
parser.add_argument(
    "output_dir", help="path to output directory to store the generated dataset"
)
parser.add_argument(
    "--limit",
    type=int,
    help="for tests purposes. Set to -1 to process everything.",
    default=10,
)
args = parser.parse_args()


SYSTEM_PROMPT = """You are an AI teacher, writing an exam out of course material.
Your task is to generate a (question, answer) pair from a given chunk from the course that is given to you.  
Return a JSON object with two keys:
- 'question': a question generated from the given chunk
- 'answer': the answer to the question
Just return the JSON, without any premamble or comment.

Chunk of the course material:
{chunk}
"""


class QAPair(BaseModel):
    question: str = Field(description="question generated from the given chunk")
    answer: str = Field(description="the answer to the question")


if __name__ == "__main__":
    assert os.path.isdir(
        args.output_dir
    ), f"Output directory not found: {args.output_dir}"
    assert os.path.isdir(args.chroma_dir), f"Chroma db not found: {args.chroma_dir}"

    load_dotenv()
    db = get_db(args.chroma_dir)
    llm = OpenAI(
        openai_api_base=os.getenv("OPENAI_BASE_URL"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name="Llama-3-70B-Instruct",
        temperature=0.0,
    )
    parser = JsonOutputParser(pydantic_object=QAPair)
    prompt = PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=["chunk"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser

    data = db.get()

    if args.limit > 0:
        n_chunks = args.limit
        output_filename = f"qa_dataset_limit={n_chunks}.csv"
    else:
        n_chunks = len(data["documents"])
        output_filename = "qa_dataset.csv"

    dataset = {"question": [], "ground_truth_answer": [], "chunk_id": []}
    for i in tqdm(range(n_chunks)):
        chunk = data["documents"][i]
        output = chain.invoke({"chunk": chunk})
        dataset["question"].append(output["question"])
        dataset["ground_truth_answer"].append(output["answer"])
        dataset["chunk_id"].append(data["ids"][i])

    df = pd.DataFrame(dataset)
    df.to_csv(str(Path(args.output_dir, output_filename)), index=False)
