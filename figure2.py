import gensim.downloader as api
from gensim.test.utils import datapath
from gensim import utils
from gensim.models import Word2Vec
from absoluteOrientation import absolute_orientation
import numpy as np
from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction
import random
random.seed(0)

def reduce_dimensions(vectors):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    points = tsne.fit_transform(vectors)
    return points

def plot(points, labels):
    import matplotlib.pyplot as plt
    N = points.shape[0] // 2
    plt.figure(figsize=(12, 12))
    # plt.subplot(211)
    X1 = [v[0] for v in points[:N]]
    Y1 = [v[1] for v in points[:N]]
    plt.scatter(X1, Y1)
    # plt.subplot(212)
    X2 = [v[0] for v in points[N:]]
    Y2 = [v[1] for v in points[N:]]
    plt.scatter(X2, Y2)

    for i in range(len(labels)):
        plt.annotate(labels[i] + '_word', (X1[i], Y1[i]))
        plt.annotate(labels[i] + '_context', (X2[i], Y2[i]))

    plt.show()

K = int(1e5)
print(api.full_information())
model = api.load('wiki-english-20171001')
api.full_information
# model = Word2Vec.load('debug.model')

wordVectors = model.wv.vectors
if hasattr(model, 'syn1'):
    contextVectors = model.syn1
else:
    contextVectors = model.syn1neg

targetWords = ['bavarian', 'kenya', 'govan', 'evesham', 'luton', 'pudding', 'mayday', 'brine']
targetWordsIndex = [model.wv.get_index(w) for w in targetWords if model.wv.has_index_for(w)]
# targetWordsIndex = random.sample(list(range(len(model.wv.index_to_key))), 10)
targetWordsVec = wordVectors[targetWordsIndex, :]
targetContextVec = contextVectors[targetWordsIndex, :]

topKWordsIndexExcludeTarget = [i for i in range(min(K, len(model.wv.index_to_key))) if model.wv.index_to_key[i] not in targetWords]

topKWordVec = wordVectors[:K, :]
topKContextVec = contextVectors[:K, :]

topKWordVecExcludeTarget = wordVectors[topKWordsIndexExcludeTarget, :]
topKContextVecExcludeTarget = contextVectors[topKWordsIndexExcludeTarget, :]

mapToWordVecSpace = absolute_orientation(topKContextVec, topKWordVec, targetContextVec)
# mapToContextVecSpace = absolute_orientation(topKWordVec, topKContextVec, targetWordsVec)

points = reduce_dimensions(np.concatenate((targetWordsVec, mapToWordVecSpace)))
plot(points, [model.wv.index_to_key[i] for i in targetWordsIndex])