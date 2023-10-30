from gensim.models import Doc2Vec, Word2Vec
 
PATH = "training/trained_models/word2vec/word2vec"
# PATH = "training/trained_models/doc2vec/doc2vecmodel.d2v"

model = Word2Vec.load(PATH)
wv = model.wv

# model = Doc2Vec.load(PATH)
# wv = model.dv

for index, word in enumerate(wv.index_to_key):
    print(f"word #{index}/{len(wv.index_to_key)} is {word}")