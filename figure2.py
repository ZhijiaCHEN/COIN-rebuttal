from math import log
import gensim.downloader as api
from gensim.test.utils import datapath
from gensim import utils
from gensim.models import Word2Vec
# from matplotlib.markers import MarkerStyle
from similarity_transform import absolute_orientation
# from absoluteOrientation import absolute_orientation
import absoluteOrientation, similarity_transform
import numpy as np
from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction
import fasttext
import random
import matplotlib.pyplot as plt
import os
import logging
import random
plt.rcParams.update({'font.size': 5})
alignFun = 1
if alignFun == 1:
    absolute_orientation = absoluteOrientation.absolute_orientation
else:
    absolute_orientation = similarity_transform.absolute_orientation

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
FIG_SIZE = 5
np.random.seed(1)
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
# contextVectors = np.random.random(size=contextVectors.shape)
# wordVectors = np.random.random(size=wordVectors.shape)

# for i in random.sample(list(range(K)), K//10):
#     wordVectors[i] = np.random.random(size=(400,))
# for i in random.sample(list(range(K)), K//10):
#     contextVectors[i] = np.random.random(size=(400,))


targetWordsIndex = [model.wv.vocab[w].index for w in targetWords]
# targetWordsIndex = random.sample(range(wordVectors.shape[0]), K)
# targetWords = [model.wv.index2word[i] for i in targetWordsIndex]

def average_sim(A, B):
    ans = 0
    for a, b in zip(A, B):
        aNorm = np.linalg.norm(a)
        bNorm = np.linalg.norm(b)
        ans += (np.inner(a, b) / (aNorm * bNorm)) * 4/(2 + aNorm / bNorm + bNorm / aNorm)
    return ans/A.shape[0]

# targetWordsIndex = random.sample(list(range(len(model.wv.index_to_key))), 10)
N = len(targetWords)
targetWordVec = wordVectors[targetWordsIndex, :]
targetContextVec = contextVectors[targetWordsIndex, :]

# wordVectors = targetWordVec
# contextVectors = targetContextVec
# targetContextVec = np.random.random(size=targetWordVec.shape)
print(f'target words average similarity: {average_sim(targetWordVec, targetContextVec)}')
# topKWordsIndexExcludeTarget = [i for i in range(min(K, len(model.wv.index_to_key))) if model.wv.index_to_key[i] not in targetWords]
def tSNE_not_align(targetWordsIndex, wordVectors, contextVectors, tSNE_Size, onlyShowTarget = True):
    targetWordVec = wordVectors[targetWordsIndex, :]
    targetContextVec = contextVectors[targetWordsIndex, :]
    topKWordVec = wordVectors[:tSNE_Size]
    topKContextVec = contextVectors[:tSNE_Size]
    avgSim = average_sim(targetWordVec, targetContextVec)
    points = reduce_dimensions(np.concatenate((targetWordVec, topKWordVec, topKContextVec, targetContextVec)))
    if not onlyShowTarget:
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

    plt.figure(figsize=(FIG_SIZE, FIG_SIZE))
    plt.scatter(X1, Y1, c='blue', marker='*')
    plt.scatter(X2, Y2, c='red', marker='*')
    for i in range(len(targetWords)):
        plt.annotate(targetWords[i] + '_word', (X1[i], Y1[i]))
        plt.annotate(targetWords[i] + '_context', (X2[i], Y2[i]))
    plt.title(f'Unaligned word and context vectors for target words. \nAverage similarity = {avgSim:.2f}.')
    plt.tight_layout()
    plt.show()

# tSNE_not_align(targetWordsIndex, wordVectors, contextVectors, K, True)

def tSNE_align_by_target_words(wordVectors, contextVectors):
    plt.figure(figsize=(FIG_SIZE, FIG_SIZE * 4))
    for i, N in enumerate([10, 100, 1000, 10000]):
        targetWordsIndex = random.sample(range(wordVectors.shape[0]), N)
        targetWordVec = wordVectors[targetWordsIndex, :]
        targetContextVec = contextVectors[targetWordsIndex, :]
        avgSimBeforeAlignment = average_sim(targetWordVec, targetContextVec)
        contextToWord = absolute_orientation(targetContextVec, targetWordVec, targetContextVec)
        avgSimAfterAlignment = average_sim(targetWordVec, contextToWord)
        logging.info(f"tSNE_align_by_target_words: Performing t-SNE for N = {N}.")
        points = reduce_dimensions(np.concatenate((targetWordVec, contextToWord)))
        X1 = [v[0] for v in points[:N]]
        Y1 = [v[1] for v in points[:N]]
        X2 = [v[0] for v in points[N:]]
        Y2 = [v[1] for v in points[N:]]
        plt.subplot(4,1,i+1)
        plt.scatter(X1, Y1, c='blue', alpha=0.3, s=5)
        plt.scatter(X2, Y2, c='red', alpha=0.3, s=5)
        plt.title(f'{N=}. Avg. similarity: {avgSimBeforeAlignment:.2f} --> {avgSimAfterAlignment:.2f}.')
    plt.tight_layout()
    plt.show()

# tSNE_align_by_target_words(wordVectors, contextVectors)

def tSNE_align_by_random_words(targetWordsIndex, wordVectors, contextVectors, showAlignWords = False):
    plt.figure(figsize=(FIG_SIZE, FIG_SIZE * 4))
    K = len(targetWordsIndex)
    for i, N in enumerate([500, 1000, 2000, 4000]):
        alignWordsIndex = targetWordsIndex
        while len(set(targetWordsIndex) & set(alignWordsIndex)) > 0:
            alignWordsIndex = random.sample(range(wordVectors.shape[0]), N)
        targetWordVec = wordVectors[targetWordsIndex, :]
        targetContextVec = contextVectors[targetWordsIndex, :]
        alignWordVec = wordVectors[alignWordsIndex, :]
        alignContextVec = contextVectors[alignWordsIndex, :]

        avgSimBeforeAlignment = average_sim(targetWordVec, targetContextVec)
        contextToWord = absolute_orientation(alignContextVec, alignWordVec, np.concatenate((alignContextVec, targetContextVec)))
        avgSimAfterAlignment = average_sim(targetWordVec, contextToWord[-K:])
        logging.info(f"tSNE_align_by_random_words: Performing t-SNE for N = {N}.")
        points = reduce_dimensions(np.concatenate((targetWordVec, alignWordVec, contextToWord)))
        plt.subplot(4,1,i+1)
        if showAlignWords:
            X1 = [v[0] for v in points[:K+N]]
            Y1 = [v[1] for v in points[:K+N]]
            X2 = [v[0] for v in points[K+N:]]
            Y2 = [v[1] for v in points[K+N:]]
            
            plt.scatter(X1, Y1, c='blue', alpha=0.3, s=5)
            plt.scatter(X2, Y2, c='red', alpha=0.3, s=5)
        
        X1 = [v[0] for v in points[:K]]
        Y1 = [v[1] for v in points[:K]]
        X2 = [v[0] for v in points[-K:]]
        Y2 = [v[1] for v in points[-K:]]
        plt.scatter(X1, Y1, c='blue', marker='*')
        plt.scatter(X2, Y2, c='red', marker='*')
        for i in range(len(targetWords)):
            plt.annotate(targetWords[i] + '_word', (X1[i], Y1[i]))
            plt.annotate(targetWords[i] + '_context', (X2[i], Y2[i]))
        plt.title(f'{N=}. Avg. similarity: {avgSimBeforeAlignment:.2f} --> {avgSimAfterAlignment:.2f}.')
    plt.tight_layout()
    plt.show()

# tSNE_align_by_random_words(targetWordsIndex, wordVectors, contextVectors, showAlignWords = True)

def tSNE_align_random_vectors():
    plt.figure(figsize=(FIG_SIZE, FIG_SIZE * 4))
    for i, N in enumerate([500, 1000, 2000, 4000]):
        fromPoints = np.random.randint(low=1, high=10, size=(N, 400))
        toPoints = np.random.randint(low=1, high=10, size=(N, 400))

        avgSimBeforeAlignment = average_sim(fromPoints, toPoints)
        alignedPoints = absolute_orientation(fromPoints, toPoints, fromPoints)
        avgSimAfterAlignment = average_sim(toPoints, alignedPoints)
        logging.info(f"tSNE_align_random_vectors: Performing t-SNE for N = {N}.")
        points = reduce_dimensions(np.concatenate((toPoints, alignedPoints)))
        plt.subplot(4,1,i+1)

        X1 = [v[0] for v in points[:points.shape[0]//2]]
        Y1 = [v[1] for v in points[:points.shape[0]//2]]
        X2 = [v[0] for v in points[points.shape[0]//2:]]
        Y2 = [v[1] for v in points[points.shape[0]//2:]]
        
        plt.scatter(X1, Y1, c='blue', alpha=0.3, s=5)
        plt.scatter(X2, Y2, c='red', alpha=0.3, s=5)
        
        plt.title(f'{N=}. Avg. similarity: {avgSimBeforeAlignment:.2f} --> {avgSimAfterAlignment:.2f}.')
    plt.tight_layout()
    plt.show()

def tSNE_align_random_vectors_by_random_vectors(showAlignPoints = True):
    plt.figure(figsize=(FIG_SIZE, FIG_SIZE * 4))
    K = 10
    for i, N in enumerate([500, 1000, 2000, 4000]):
        fromPointsTrain = np.random.randint(low=1, high=10, size=(N, 400))
        toPointsTrain = np.random.randint(low=1, high=10, size=(N, 400))
        fromPointsTest = np.random.randint(low=1, high=10, size=(K, 400))
        toPointsTest = np.random.randint(low=1, high=10, size=(K, 400))


        avgSimBeforeAlignment = average_sim(fromPointsTest, toPointsTest)
        alignedPoints = absolute_orientation(fromPointsTrain, toPointsTrain, np.concatenate((fromPointsTest, fromPointsTrain)))
        avgSimAfterAlignment = average_sim(toPointsTest, alignedPoints[:K])
        logging.info(f"tSNE_align_random_vectors: Performing t-SNE for N = {N}.")
        points = reduce_dimensions(np.concatenate((alignedPoints, toPointsTrain, toPointsTest)))
        plt.subplot(4,1,i+1)
        if showAlignPoints:
            X1 = [v[0] for v in points[:K+N]]
            Y1 = [v[1] for v in points[:K+N]]
            X2 = [v[0] for v in points[K+N:]]
            Y2 = [v[1] for v in points[K+N:]]
            
            plt.scatter(X1, Y1, c='blue', alpha=0.3, s=5)
            plt.scatter(X2, Y2, c='red', alpha=0.3, s=5)

        X1 = [v[0] for v in points[:K]]
        Y1 = [v[1] for v in points[:K]]
        X2 = [v[0] for v in points[-K:]]
        Y2 = [v[1] for v in points[-K:]]
        plt.scatter(X1, Y1, c='blue', marker='*')
        plt.scatter(X2, Y2, c='red', marker='*')
        plt.title(f'{N=}. Avg. similarity: {avgSimBeforeAlignment:.2f} --> {avgSimAfterAlignment:.2f}.')
    plt.tight_layout()
    plt.show()

tSNE_align_random_vectors_by_random_vectors()
exit()


print('Mapping context vector into word vector...')
contextInWordSpace = absolute_orientation(topKContextVec, topKWordVec, np.concatenate((topKContextVec, targetContextVec)))
print('Mapping word vector into context vector...')
wordInContextSpace = absolute_orientation(topKWordVec, topKContextVec, np.concatenate((topKWordVec, targetWordVec)))
print(f'top words average similarity after alignment: {average_sim(topKWordVec, contextInWordSpace[:K])}')
plt.figure(figsize=(6, 12))
plt.subplot(211)
print('Performaning t-SNE on word vector space...')
wordSpacePoints = reduce_dimensions(np.concatenate((targetWordVec, topKWordVec, contextInWordSpace)))
wordSpacePoints = reduce_dimensions(np.concatenate((targetWordVec, contextInWordSpace[K:])))
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

# for i in range(len(targetWords)):
#     plt.annotate(targetWords[i] + '_word', (X1[i], Y1[i]))
#     plt.annotate(targetWords[i] + '_context', (X2[i], Y2[i]))

plt.subplot(212)
print('Performaning t-SNE on context vector space...')
contextSpacePoints = reduce_dimensions(np.concatenate((targetContextVec, topKContextVec, wordInContextSpace)))
contextSpacePoints = reduce_dimensions(np.concatenate((targetContextVec, wordInContextSpace[K:])))
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

# for i in range(len(targetWords)):
#     plt.annotate(targetWords[i] + '_context', (X1[i], Y1[i]))
#     plt.annotate(targetWords[i] + '_word', (X2[i], Y2[i]))
plt.title('t-SNE in Context Vector Space')
plt.show()