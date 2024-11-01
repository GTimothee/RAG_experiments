from typing import List

import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from whoosh.index import open_dir
from whoosh.qparser import MultifieldParser
from whoosh import scoring
from whoosh import qparser
from whoosh.lang.porter import stem

from langchain_core.documents import Document


class KeywordRetriever:

    def __init__(self, index_dirpath: str, k=4):
        self.k = k
        self.stop_words = set(stopwords.words("english"))
        self.stop_words.add("?")
        self.ix = open_dir(index_dirpath)
        self.parser = MultifieldParser(
            ["content"],
            self.ix.schema,
            group=qparser.OrGroup.factory(0.9),
        )

    def invoke(self, query: str) -> List[Document]:
        print(f"Input query: {query}")
        
        with self.ix.searcher(weighting=scoring.BM25F()) as searcher:
            
            formatted_query = self._process_query(query)

            results = searcher.search(formatted_query, limit=self.k)
            return [
                Document(
                    metadata=result["metadata"], 
                    page_content=result["text_content"]
                )
                for result in results
            ]

    def _process_query(self, query):
        # tokenize 
        word_tokens = word_tokenize(query)
        print(f"tokenized query: {word_tokens}")

        # stem and filter out stop words
        filtered_query = " ".join(
            [
                stem(word)
                for word in word_tokens
                if word.lower() not in self.stop_words
            ]
        )
        print(f"stemmed-filtered_query: {filtered_query}")

        parsed_query = self.parser.parse(filtered_query)
        print(f"parsed_query: {parsed_query}")
        return parsed_query