import os
import argparse
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

parser = argparse.ArgumentParser()
parser.add_argument('chroma_dir', help="path to chroma db")
parser.add_argument('dataset_filepath', help="dataset in csv format")
parser.add_argument('answers_filepath', help="answers to dataset generated with RAG, in csv format")
args = parser.parse_args()


SYSTEM_PROMPT = """You are a top-tier grading software belonging to a school.
Your task is to give a grade to evaluate the answer goodness to a given question, given the ground truth answer.

You will be given a piece of data containing: 
- a 'question'
- an 'answer': the answer to the question from the student
- a 'ground truth answer': the expected answer to the question

Provide your answer as a JSON with two keys: 
- 'completeness': A float between 0 and 1. The percentage of the ground truth answer that is present in the student's answer. A score of 1 means that all the information in the 'ground truth answer' can be found in the 'answer'. No matter if the answer contains more information than expected. A score of 0 means that no information present in the 'ground truth answer' can be found in the 'answer'.
- 'conciseness': A float between 0 and 1. The percentage of the answer that is part of the ground truth. Conciseness measures how much of the answer is really useful.

Here is the data to evaluate: 
- 'question': {question}
- 'answer': {answer}
- 'ground truth answer': {ground_truth_answer}

Provide your answer as a JSON, with no additional text.
"""


class Evaluation(BaseModel):
    completeness: float = Field(description="A float between 0 and 1. The percentage of the ground truth answer that is present in the student's answer. A score of 1 means that all the information in the 'ground truth answer' can be found in the 'answer'. No matter if the answer contains more information than expected. A score of 0 means that no information present in the 'ground truth answer' can be found in the 'answer'.")
    conciseness: float = Field(description="A float between 0 and 1. The percentage of the answer that is part of the ground truth. Conciseness measures how much of the answer is really useful.")


if __name__ == "__main__":

    load_dotenv()

    df = pd.concat([
        pd.read_csv(args.dataset_filepath),
        pd.read_csv(args.answers_filepath)
    ], axis=1)

    llm = OpenAI(
        openai_api_base=os.getenv("OPENAI_BASE_URL"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name="Llama-3-70B-Instruct",
        temperature=0.0,
    )
    parser = JsonOutputParser(pydantic_object=Evaluation)
    prompt = PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=["question", "answer", "ground_truth_answer"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser

    conciseness, completeness = 0., 0.
    ranks = []
    for row in tqdm(df.itertuples(), total=len(df), desc='Evaluating answers...'):
        output = chain.invoke({
            "question": row.question,
            "answer": row.answers,
            "ground_truth_answer": row.ground_truth_answer
        })
        completeness += output['completeness']
        conciseness += output['conciseness']
        ranks.append(row.ranks)
    
    mean_conciseness = conciseness / len(df)
    mean_completeness = completeness / len(df)

    print({
        "mean_completeness": f"{round(mean_completeness*100)} %",
        "mean_conciseness": f"{round(mean_conciseness*100)} %"
    })

    print(pd.Series(ranks).value_counts())
