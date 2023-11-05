import logging

import pandas as pd
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

MODEL_OUTPUT_FILE = 'word2vecmodel.w2v'
EPOCHS = 100
VECTOR_SIZE = 300
WINDOW_SIZE = 15
MIN_COUNT = 20
SAMPLE = 1e-5
NEGATIVE_SAMPLES = 5
USE_SKIP_GRAM = 1
WORKERS = 4

class WikipediaCorpus:
  def __init__(self, data_file: str):
    self.data = pd.read_pickle(data_file)
    self.stopwords = stopwords.words('english')

  def __iter__(self):
    for _, row in self.data.iterrows():
      text = row['text'].split()
      text = [token for token in text if token not in self.stopwords]

      yield text


def train_word2vec(
    corpus: str,
    epochs: int = EPOCHS,
    vector_size: int = VECTOR_SIZE,
    window: int = WINDOW_SIZE,
    min_count: int = MIN_COUNT,
    sample: int = SAMPLE,
    workers: int = WORKERS,
    negative: int = NEGATIVE_SAMPLES,
    sg: int = USE_SKIP_GRAM):
  
  model = Word2Vec(
    vector_size = vector_size, 
    window = window,
    min_count = min_count,
    sample = sample, 
    workers = workers, 
    negative = negative,
    sg = sg
  )
  model.build_vocab(corpus_file = corpus)

  model.train(corpus_file = corpus, total_examples=model.corpus_count, total_words=model.corpus_total_words, epochs=epochs)

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
  parser.add_argument('--negative-samples', type=int, default=NEGATIVE_SAMPLES, metavar='NEGATIVE_SAMPLES', help='Number of negative samples to use')
  parser.add_argument('--sg', type=int, default=USE_SKIP_GRAM, metavar='USE_SKIP_GRAM', help='Whether to use skip-gram')
  args = parser.parse_args()

  # corpus = WikipediaCorpus(args.data)

  model = train_word2vec(
    args.data,
    epochs = args.epochs,
    vector_size = args.vector_size,
    window = args.window_size,
    min_count = args.min_count,
    sample = args.sample,
    workers = args.workers,
    negative = args.negative_samples,
    sg = args.sg
  )

  model.save(args.output)