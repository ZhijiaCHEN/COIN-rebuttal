import gensim.downloader as api
from gensim.test.utils import datapath
from gensim import utils
from gensim.models import Word2Vec
from matplotlib.markers import MarkerStyle
from absoluteOrientation import absolute_orientation
import numpy as np
from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction
import fasttext
import random
import matplotlib.pyplot as plt
import os
import logging
import random
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# random.seed(0)

def reduce_dimensions(vectors):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    points = tsne.fit_transform(vectors)
    return points


K = int(1e3)
targetWords = ['bavarian', 'kenya', 'govan', 'evesham', 'luton', 'pudding', 'mayday', 'brine', 'sunglasses', 'patchwork']
print('Loading saved word vectors and context vectors...')

model = Word2Vec.load('wiki.model')
contextVectors = model.trainables.syn1neg
wordVectors = model.wv.syn0
# for i in random.sample(list(range(K)), K//10):
#     wordVectors[i] = np.random.random(size=(400,))
# for i in random.sample(list(range(K)), K//10):
#     contextVectors[i] = np.random.random(size=(400,))


targetWordsIndex = [model.wv.vocab[w].index for w in targetWords]


def average_sim(A, B):
    ans = 0
    for a, b in zip(A, B):
        ans += np.inner(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))
    return ans/A.shape[0]

# targetWordsIndex = random.sample(list(range(len(model.wv.index_to_key))), 10)
N = len(targetWords)
targetWordVec = wordVectors[targetWordsIndex, :]
targetContextVec = contextVectors[targetWordsIndex, :]
# targetContextVec = np.random.random(size=targetWordVec.shape)
print(f'target words average similarity: {average_sim(targetWordVec, targetContextVec)}')
# topKWordsIndexExcludeTarget = [i for i in range(min(K, len(model.wv.index_to_key))) if model.wv.index_to_key[i] not in targetWords]

topKWordVec = wordVectors[:K]
topKContextVec = contextVectors[:K]
print(f'top words average similarity before alignment: {average_sim(topKWordVec, topKContextVec)}')
plt.figure(figsize=(6, 6))
print('Performaning t-SNE on unaligned word and context vectors...')
points = reduce_dimensions(np.concatenate((targetWordVec, wordVectors[:K], contextVectors[:K], targetContextVec)))
X1 = [v[0] for v in points[:points.shape[0]//2]]
Y1 = [v[1] for v in points[:points.shape[0]//2]]
X2 = [v[0] for v in points[points.shape[0]//2:]]
Y2 = [v[1] for v in points[points.shape[0]//2:]]
plt.scatter(X1, Y1, c='blue', alpha=0.3, s=5)
plt.scatter(X2, Y2, c='red', alpha=0.3, s=5)

X1 = [v[0] for v in points[:N]]
Y1 = [v[1] for v in points[:N]]

X2 = [v[0] for v in points[-N:]]
Y2 = [v[1] for v in points[-N:]]
plt.scatter(X1, Y1, c='blue', marker='*')
plt.scatter(X2, Y2, c='red', marker='*')
for i in range(len(targetWords)):
    plt.annotate(targetWords[i] + '_word', (X1[i], Y1[i]))
    plt.annotate(targetWords[i] + '_context', (X2[i], Y2[i]))
plt.title('t-SNE on unaligned word and context vectors')
plt.show()

print('Mapping context vector into word vector...')
contextInWordSpace = absolute_orientation(topKContextVec, topKWordVec, np.concatenate((topKContextVec, targetContextVec)))
print('Mapping word vector into context vector...')
wordInContextSpace = absolute_orientation(topKWordVec, topKContextVec, np.concatenate((topKWordVec, targetWordVec)))
print(f'top words average similarity after alignment: {average_sim(topKWordVec, contextInWordSpace[:K])}')
plt.figure(figsize=(6, 12))
plt.subplot(211)
print('Performaning t-SNE on word vector space...')
wordSpacePoints = reduce_dimensions(np.concatenate((targetWordVec, topKWordVec, contextInWordSpace)))
X1 = [v[0] for v in wordSpacePoints[:wordSpacePoints.shape[0]//2]]
Y1 = [v[1] for v in wordSpacePoints[:wordSpacePoints.shape[0]//2]]
X2 = [v[0] for v in wordSpacePoints[wordSpacePoints.shape[0]//2:]]
Y2 = [v[1] for v in wordSpacePoints[wordSpacePoints.shape[0]//2:]]
plt.scatter(X1, Y1, c='blue', alpha=0.3, s=5)
plt.scatter(X2, Y2, c='red', alpha=0.3, s=5)

X1 = [v[0] for v in wordSpacePoints[:N]]
Y1 = [v[1] for v in wordSpacePoints[:N]]
plt.scatter(X1, Y1, c='blue', marker='*')

X2 = [v[0] for v in wordSpacePoints[-N:]]
Y2 = [v[1] for v in wordSpacePoints[-N:]]
plt.scatter(X2, Y2, c='red', marker='*')
plt.title('t-SNE in Word Vector Space')

for i in range(len(targetWords)):
    plt.annotate(targetWords[i] + '_word', (X1[i], Y1[i]))
    plt.annotate(targetWords[i] + '_context', (X2[i], Y2[i]))

plt.subplot(212)
print('Performaning t-SNE on context vector space...')
contextSpacePoints = reduce_dimensions(np.concatenate((targetContextVec, topKContextVec, wordInContextSpace)))
X1 = [v[0] for v in contextSpacePoints[:contextSpacePoints.shape[0]//2]]
Y1 = [v[1] for v in contextSpacePoints[:contextSpacePoints.shape[0]//2]]
X2 = [v[0] for v in contextSpacePoints[contextSpacePoints.shape[0]//2:]]
Y2 = [v[1] for v in contextSpacePoints[contextSpacePoints.shape[0]//2:]]
plt.scatter(X1, Y1, c='blue', alpha=0.3, s=5)
plt.scatter(X2, Y2, c='red', alpha=0.3, s=5)

X1 = [v[0] for v in contextSpacePoints[:N]]
Y1 = [v[1] for v in contextSpacePoints[:N]]
plt.scatter(X1, Y1, c='blue', marker='*')

X2 = [v[0] for v in contextSpacePoints[-N:]]
Y2 = [v[1] for v in contextSpacePoints[-N:]]
plt.scatter(X2, Y2, c='red', marker='*')

for i in range(len(targetWords)):
    plt.annotate(targetWords[i] + '_context', (X1[i], Y1[i]))
    plt.annotate(targetWords[i] + '_word', (X2[i], Y2[i]))
plt.title('t-SNE in Context Vector Space')
plt.show()