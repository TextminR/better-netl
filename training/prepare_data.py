import logging
import sys
from tqdm import tqdm
from argparse import ArgumentParser

import pandas as pd
from datasets import load_dataset

from nltk.parse import CoreNLPParser

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%H:%M:%S%p')

CORENLP_SERVER_URL = 'http://localhost:9000'
OUTPUT_FILE = 'wiki_tokenized.pickle'

parser = ArgumentParser()
parser.add_argument('subset', type=str, help='Subset of the wikipedia dataset to use (e.g. "en", "simple")')
parser.add_argument('--server', type=str, default=CORENLP_SERVER_URL, help='URL of the CoreNLP server to use (default: %(default)s)')

def load_data(subset: str):
  logging.info('Loading data')
  return load_dataset('wikipedia', f'20220301.{subset}', split='train').to_pandas().drop(columns=['url'])

def tokenize(data: pd.DataFrame, server_url: str):
  parser = CoreNLPParser(url = server_url)
  
  logging.info('Tokenizing data')
  data['text'] = data['text'].str.lower().replace('\n', ' ')

  for i in tqdm(range(len(data))):
    try:
      data['text'][i] = ' '.join(parser.tokenize(data['text'][i].strip()))
    except KeyboardInterrupt:
      sys.exit()
    except:
      logging.warning(f'Failed to tokenize text with id {data["id"][i]}. Skipping.')

  return data

if __name__ == '__main__':
  args = parser.parse_args()

  data = load_data(args.subset)
  data = tokenize(data, args.server)
  
  logging.info('Saving...')
  data.to_pickle(OUTPUT_FILE)

