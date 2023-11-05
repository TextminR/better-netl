from argparse import ArgumentParser

import pandas as pd
import numpy as np
from numpy.linalg import norm

from gensim.models import Doc2Vec
from gensim.models import Word2Vec
from gensim import matutils

parser = ArgumentParser()
parser.add_argument('doc2vec_path', type=str, help='Path to the doc2vec model')
parser.add_argument('word2vec_path', type=str, help='Path to the word2vec model')
parser.add_argument('topics', type=str, help='Path to the topic file')

def cosine_similarity(a: np.array, b: np.array):
  return np.dot(a, b) / (norm(a) * norm(b))

def normalize(a: np.array):
  return a / norm(a)

def get_labels(doc2vec, word2vec, 
               topics: dict):

  doc2vec_dv_norms = normalize(doc2vec.dv.vectors)
  doc2vec_wv_norms = normalize(doc2vec.wv.vectors)
  word2vec_wv_norms = normalize(word2vec.wv.vectors)

  for topic in topics.values():
    valdoc2vec = 0.0
    valword2vec = 0.0

    for item in topic:
      try:
          tempdoc2vec = doc2vec_wv_norms[doc2vec.wv.key_to_index[item]]
      except:
          pass
      else:
          meandoc2vec = normalize(tempdoc2vec)
          distsdoc2vec = np.dot(doc2vec_dv_norms, meandoc2vec)
          valdoc2vec = valdoc2vec + distsdoc2vec
       
      try:
          tempword2vec = word2vec_wv_norms[word2vec.wv.key_to_index[item]]
      except:
          pass
      else:
          meanword2vec = normalize(tempword2vec)
          distsword2vec = np.dot(word2vec_wv_norms, meanword2vec)
          valword2vec = valword2vec + distsword2vec
      
    avgdoc2vec = valdoc2vec/float(len(topic))
    avgword2vec = valword2vec/float(len(topic))

    bestdoc2vec = matutils.argsort(avgdoc2vec, topn=10, reverse=True)
    resultdoc2vec = [doc2vec.dv.index_to_key[sim] for sim in bestdoc2vec]

    bestword2vec = matutils.argsort(avgword2vec, topn=10, reverse=True)
    resultword2vec = [word2vec.wv.index_to_key[sim] for sim in bestword2vec]

    yield [resultdoc2vec, resultword2vec]

if __name__ == '__main__':
  args = parser.parse_args()

  doc2vec = Doc2Vec.load(args.doc2vec_path)
  word2vec = Word2Vec.load(args.word2vec_path)

  topics = pd.read_csv(args.topics)
  topic_list = topics.set_index('topic_id').T.to_dict('list')
  
  for result in get_labels(doc2vec, word2vec, topic_list):
    print(result)