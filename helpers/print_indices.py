from argparse import ArgumentParser
from gensim.models import Doc2Vec, Word2Vec

parser = ArgumentParser(description='Get indices of words in a word2vec or doc2vec model')
parser.add_argument('model_path', help='path to the model')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--word2vec', help='get indices of words in a word2vec model', action='store_true')
group.add_argument('--doc2vec', help='get indices of words in a doc2vec model', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    
    model = Word2Vec.load(args.model_path) if args.word2vec else Doc2Vec.load(args.model_path)
    vectors = model.wv if args.word2vec else model.dv

    for index, word in enumerate(vectors.index_to_key):
        print(f"word #{index}/{len(vectors.index_to_key)} is {word}")
