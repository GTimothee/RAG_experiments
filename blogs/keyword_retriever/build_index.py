import argparse
from pathlib import Path
from dotenv import load_dotenv

from whoosh.fields import Schema, TEXT, STORED
from whoosh.index import create_in
from whoosh.analysis import StemmingAnalyzer

from chromadb_iterator import database_iterator


parser = argparse.ArgumentParser()
parser.add_argument(
    "chroma_dir", help="Input dirpath containing the chroma db files"
)
parser.add_argument(
    "output_dirpath", help="Output dirpath of the index to be created"
)
args = parser.parse_args()


if __name__ == "__main__":

    load_dotenv()

    index_dirpath = Path(args.output_dirpath)
    if not index_dirpath.exists():
        print("creating output dirpath...")
        index_dirpath.mkdir(parents=True)
        print("done.")

    # create the index
    stem_ana = StemmingAnalyzer()
    schema = Schema(
        content=TEXT(analyzer=stem_ana),
        text_content=STORED,
        metadata=STORED,
    )

    # fill in the db
    ix = create_in(str(index_dirpath), schema)
    writer = ix.writer()

    total = 0
    for batch in database_iterator('data/chroma_db_1000'):
        batch_size = len(batch["documents"])
        total += batch_size 

        for i in range(batch_size):
            doc = batch["documents"][i]
            meta = batch["metadatas"][i]

            writer.add_document(
                text_content=doc,
                content=doc,
                metadata=meta
            )

    # save
    print("Saving...")
    writer.commit()
    print("=> Ingested {} documents.".format(total))
    print("Done.")
