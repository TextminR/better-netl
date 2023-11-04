import logging
from argparse import ArgumentParser

import pandas as pd

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%H:%M:%S%p')

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

def process_corpus(data_file: str):
  data = pd.read_pickle(data_file)

  corpus = []
  for _, row in data.iterrows():
    corpus.append(TaggedDocument(row['text'].split(), [row['title'].replace(' ', '_')]))

  return corpus

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

if __name__ == '__main__':
  args = parser.parse_args()

  corpus = process_corpus(args.data)

  model = Doc2Vec(
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
  model.build_vocab(corpus)

  model.train(corpus, total_examples=model.corpus_count, epochs=args.epochs)
  model.save(args.output)