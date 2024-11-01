import argparse
import pandas as pd

from pathlib import Path 
from dotenv import load_dotenv
from tqdm import tqdm

from library.rag import get_rag_chain_eval


parser = argparse.ArgumentParser()
parser.add_argument('chroma_dir', help="path to chroma db")
parser.add_argument('dataset_filepath', help="dataset in csv format")
parser.add_argument(
    "output_dir", help="path to output directory to store the generated dataset"
)
args = parser.parse_args()


if __name__ == "__main__":
    load_dotenv()
    rag_chain, retriever, db = get_rag_chain_eval(chroma_db_dirpath="data/chroma_db_1000")
    df = pd.read_csv(args.dataset_filepath)
    outputs = {
        'answers': [], 'ranks': []
    }

    for row in tqdm(df.itertuples(), total=len(df), desc='Generating answers...'):
        documents = retriever.invoke(row.question)

        # generate answer
        output = rag_chain.invoke({
            "question": row.question,
            "context": '\n'.join([doc.page_content for doc in documents])
        })
        outputs['answers'].append(output)

        # compute rank of the target documents in the list of retrieved documents
        target_chunk = db.get(row.chunk_id)['documents'][0]
        rank = None
        for i, chunk in enumerate(documents):
            if chunk.page_content == target_chunk:
                rank = i
        outputs['ranks'].append(rank)

    pd.DataFrame(outputs).to_csv(
        str(Path(args.output_dir, f"{Path(args.dataset_filepath).stem}" + "_answers.csv")), index=False)
    
    
    