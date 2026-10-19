import bm25s
import BM25

corpus = BM25.load("vllm-0.10.1/*.txt")

corpus_tokens = bm25s.tokenize(corpus, stopwords="en")

retriever = bm25s.BM25()
retriever.index(corpus_tokens)


query = "dogs"
query_tokens = bm25s.tokenize(query)

results, scores = retriever.retrieve(query_tokens, corpus=corpus, k=2)


print(results)
[['cats and dogs are pets.'
'cats and dogs are pet animals though I prefer dogs. Dogs obey our commands, can be trained easily and play with us all the time.']]

print(scores)
