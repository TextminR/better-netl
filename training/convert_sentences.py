import logging

import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

INPUT_FILE = 'wiki_tokenized.pickle'
OUTPUT_FILE = 'wiki_sentences.txt'

if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%H:%M:%S%p')
  
  data = pd.read_pickle(INPUT_FILE)#.sample(n=1000)
  stop_words = stopwords.words('english')

  with open(OUTPUT_FILE, 'w+') as f:
    for _, row in data.iterrows():
      for sentence in nltk.sent_tokenize(row['text']):
        words = sentence.split()
        words = [word for word in words if word not in stop_words]
        if len(words) > 0:
          f.write(' '.join(words) + '\n')