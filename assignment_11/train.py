import sys
import numpy as np
import pandas as pd

from nltk import RegexpTokenizer
from gensim.models import Word2Vec

tokenizer = RegexpTokenizer(r"\w+")

data_df = pd.read_csv(sys.argv[1])

text_data = data_df['text'].values
labels = data_df['label'].values

fake_texts = text_data[np.where(labels == 'FAKE')]
fake_texts = np.array(list(map(lambda x:x.lower(), fake_texts)))
fake_texts = np.array(list(map(tokenizer.tokenize, fake_texts)))


real_texts = text_data[np.where(labels == 'REAL')]
real_texts = np.array(list(map(lambda x:x.lower(), real_texts)))
real_texts = np.array(list(map(tokenizer.tokenize, real_texts)))


fake_model = Word2Vec(fake_texts, size=100, window=10, min_count=5, workers=5)
real_model = Word2Vec(real_texts, size=100, window=10, min_count=5, workers=5)

fake_model.save(sys.argv[2]+'/fake.model')
real_model.save(sys.argv[2]+'/real.model')