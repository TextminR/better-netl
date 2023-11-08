import logging

import re
import pandas as pd

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

MODEL_OUTPUT_FILE = 'doc2vecmodel.d2v'
EPOCHS = 20
VECTOR_SIZE = 300
WINDOW_SIZE = 15
MIN_COUNT = 20
SAMPLE = 1e-5
HIERARCHIAL_SOFTMAX = 0
DISTRIBUTED_MEMORY = 0
NEGATIVE_SAMPLES = 5
DISTRIBUTED_BAG_OF_WORDS = 1
DM_CONCATENATION = 0
WORKERS = 4

class WikipediaCorpus:
  def __init__(self, data_file: str):
    self.data = pd.read_pickle(data_file)

  def __iter__(self):
    for _, row in self.data.iterrows():
      title = row['title']

      if not re.findall('\(.*\)', title):
        yield TaggedDocument(row['text'].split(), [title.replace(' ', '_')])

def train_doc2vec(
    corpus: WikipediaCorpus,
    epochs: int = EPOCHS,
    vector_size: int = VECTOR_SIZE,
    window: int = WINDOW_SIZE,
    min_count: int = MIN_COUNT,
    sample: int = SAMPLE,
    workers: int = WORKERS,
    hs: int = HIERARCHIAL_SOFTMAX,
    dm: int = DISTRIBUTED_MEMORY,
    negative: int = NEGATIVE_SAMPLES,
    dbow_words: int = DISTRIBUTED_BAG_OF_WORDS,
    dm_concat: int = DM_CONCATENATION):
  
  model = Doc2Vec(
    vector_size=vector_size, 
    window=window,
    min_count=min_count,
    sample=sample, 
    workers=workers, 
    hs=hs,
    dm=dm,
    negative=negative,
    dbow_words=dbow_words,
    dm_concat=dm_concat
  )
  model.build_vocab(corpus)

  model.train(corpus, total_examples=model.corpus_count, epochs=epochs)

  return model

if __name__ == '__main__':
  from argparse import ArgumentParser

  logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%H:%M:%S%p')

  parser = ArgumentParser()
  parser.add_argument('data', type=str, help='Path to the tokenized data file')
  parser.add_argument('--output', '-o', type=str, default=MODEL_OUTPUT_FILE, metavar='OUTPUT_FILE', help='Path to the model output file')
  parser.add_argument('--epochs', type=int, default=EPOCHS, metavar='EPOCHS', help='Number of epochs to train the model')
  parser.add_argument('--vector-size', type=int, default=VECTOR_SIZE, metavar='VECTOR_SIZE', help='Size of the word vectors')
  parser.add_argument('--window-size', type=int, default=WINDOW_SIZE, metavar='WINDOW_SIZE', help='Window size')
  parser.add_argument('--min-count', type=int, default=MIN_COUNT, metavar='MIN_COUNT', help='Minimum number of times a word must appear to be included in the vocabulary')
  parser.add_argument('--sample', type=float, default=SAMPLE, metavar='SAMPLE', help='Threshold for configuring which higher-frequency words are randomly downsampled')
  parser.add_argument('--workers', type=int, default=WORKERS, metavar='WORKERS', help='Number of worker threads to train the model')
  parser.add_argument('--hierarchial-softmax', type=int, default=HIERARCHIAL_SOFTMAX, metavar='HIERARCHIAL_SOFTMAX', help='Whether to use hierarchial softmax')
  parser.add_argument('--distributed-memory', type=int, default=DISTRIBUTED_MEMORY, metavar='DISTRIBUTED_MEMORY', help='Whether to use distributed memory')
  parser.add_argument('--negative-samples', type=int, default=NEGATIVE_SAMPLES, metavar='NEGATIVE_SAMPLES', help='Number of negative samples to use')
  parser.add_argument('--dbow', type=int, default=DISTRIBUTED_BAG_OF_WORDS, metavar='DISTRIBUTED_BAG_OF_WORDS', help='Whether to use distributed bag of words')
  parser.add_argument('--dm-concat', type=int, default=DM_CONCATENATION, metavar='DM_CONCATENATION', help='Whether to concatenate the input and output vectors in distributed memory')
  args = parser.parse_args()

  corpus = WikipediaCorpus(args.data)

  model = train_doc2vec(
    corpus,
    epochs=args.epochs,
    vector_size=args.vector_size, 
    window=args.window_size,
    min_count=args.min_count,
    sample=args.sample, 
    workers=args.workers, 
    hs=args.hierarchial_softmax,
    dm=args.distributed_memory,
    negative=args.negative_samples,
    dbow_words=args.dbow,
    dm_concat=args.dm_concat
  )

  model.save(args.output)