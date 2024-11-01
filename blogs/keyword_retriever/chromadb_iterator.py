import chromadb
from chromadb import Settings
from dotenv import load_dotenv
from library.database import get_db


def database_iterator(database_dirpath: str, batch_size: int = 100):
    client = get_db(database_dirpath)
    offset = 0
    while offset < client._collection.count():
        print(f"offset={offset}")
        batch = client.get(offset=offset, limit=batch_size)
        offset += len(batch["documents"])
        yield batch


if __name__ == "__main__":
    load_dotenv()
    n = 0
    for batch in database_iterator('data/chroma_db_1000'):
        nb = len(batch['documents'])
        print(f"yields batch of size: {nb}")
        n += nb
    print(f"Retrieved {n} documents.")