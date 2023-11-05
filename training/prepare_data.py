import logging
import sys
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from datasets import load_dataset

from nltk.parse import CoreNLPParser

CORENLP_SERVER_URL = 'http://localhost:9000'
OUTPUT_FILE = 'wiki_tokenized.pickle'

def tokenize_wiki(subset: str, parser_url: str = CORENLP_SERVER_URL):
  logging.info('Loading data')
  data = load_dataset('wikipedia', f'20220301.{subset}', split='train').to_pandas().drop(columns=['url'])

  parser = CoreNLPParser(url = parser_url)
  
  logging.info('Tokenizing data')
  data['title'] = data['title'].str.lower()
  data['text'] = data['text'].str.lower()

  with logging_redirect_tqdm():
    for i in tqdm(range(len(data))):
      try:
        data['text'][i] = ' '.join(parser.tokenize(data['text'][i].strip()))
      except KeyboardInterrupt:
        sys.exit()
      except:
        logging.warning(f'Failed to tokenize text with id {data["id"][i]}. Skipping.')

  return data

if __name__ == '__main__':
  from argparse import ArgumentParser

  logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%H:%M:%S%p')

  parser = ArgumentParser()
  parser.add_argument('subset', type=str, help='Subset of the wikipedia dataset to use (e.g. "en", "simple")')
  parser.add_argument('--parser', type=str, metavar='URL', default=CORENLP_SERVER_URL, help='URL of the CoreNLP server to use (default: %(default)s)')
  parser.add_argument('--output', '-o', type=str, metavar='OUTPUT_FILE', default=OUTPUT_FILE, help='Path to the output file (default: %(default)s)')
  args = parser.parse_args()
  
  data = tokenize_wiki(args.subset, args.parser)
  
  logging.info('Saving...')
  data.to_pickle(args.output)

