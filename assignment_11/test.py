import sys
import numpy as np

from gensim.models import Word2Vec

real_model = Word2Vec.load(sys.argv[1]+'/real.model')
fake_model = Word2Vec.load(sys.argv[1]+'/fake.model')

words = open(sys.argv[2]).readline()
words = words.strip().split(',')
words = list(map(lambda x:x.lower(), words))


print("\n fake Model\n")

for q_word in words:

    print("\nSimilar words for {}:\n".format(q_word))

    fake_sim = []
    fake_vocab = np.array(list(fake_model.wv.vocab.keys()))

    for fake_word in fake_vocab:

        cos_sim = fake_model.wv.similarity(fake_word, q_word)

        fake_sim.append(cos_sim)

    fake_sim = np.array(fake_sim)

    top_index = fake_sim.argsort()[-6:-1][::-1]

    for word in fake_vocab[top_index]:
        print(word)


print("\n Real Model\n")

for q_word in words:

    print("\nSimilar words for {}: \n".format(q_word))

    real_sim = []
    real_vocab = np.array(list(real_model.wv.vocab.keys()))

    for real_word in real_vocab:

        cos_sim = real_model.wv.similarity(real_word, q_word)

        real_sim.append(cos_sim)

    real_sim = np.array(real_sim)

    top_index = real_sim.argsort()[-6:-1][::-1]

    for word in real_vocab[top_index]:
        print(word)
