# RAG_experiments

## Setup the DB
1. Download the dataset
```wget https://huggingface.co/datasets/m-ric/huggingface_doc/resolve/main/huggingface_doc.csv```
2. Build the db
```python build_vector_database.py```

## Blog posts

### RAG evaluation

1. Generate the dataset
```python test_procedure_for_rag/generate_qa_pairs.py data/chroma_db_1000/ data/```
2. Generate the answers using the RAG
```python test_procedure_for_rag/generate_answers.py data/chroma_db_1000/ data/qa_dataset_limit\=10.csv data/```