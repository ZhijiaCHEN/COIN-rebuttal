from math import log
import gensim.downloader as api
from gensim.test.utils import datapath
from gensim import utils
from gensim.models import Word2Vec
# from matplotlib.markers import MarkerStyle

# the first and the second transformation function
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
from pca import pca
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
plt.rcParams.update({'font.size': 8})

# the third transformation function
def linear_transform(fromPoints, toPoints, target):
    M = np.linalg.pinv(fromPoints).dot(toPoints)
    return target.dot(M)

alignFun = 3
if alignFun == 1:
    logging.info("Align with absolute orientation 2020.")
    absolute_orientation = absoluteOrientation.absolute_orientation
elif alignFun == 2:
    logging.info("Align with absolute orientation 1995.")
    absolute_orientation = similarity_transform.absolute_orientation
else:
    logging.info("Align with linear transformation.")
    absolute_orientation = linear_transform

# set figure size
FIG_SIZE = 4
GraphSize = (2 * FIG_SIZE, 2 * FIG_SIZE)
np.random.seed(1)
random.seed(5)

# reduce the vectors space to 2-dimensional space so we can see how well the word vectors and context vectors are aligned.
def reduce_dimensions(vectors):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)
    vectors = pca(vectors)
    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, perplexity=10)
    points = tsne.fit_transform(vectors)
    return points

targetWords = ['bavarian', 'kenya', 'govan', 'evesham', 'luton', 'pudding', 'mayday', 'brine', 'sunglasses', 'patchwork']
logging.info('Loading saved word vectors and context vectors...')

# load pretrain wiki model, the latest version of gensim you can use is 3.8.3, so your gensim version should be something 3.*.* and <= 3.8.3
model = Word2Vec.load('wiki.model')
contextVectors = model.trainables.syn1neg # each row is a context vector, sorted by word frequency
wordVectors = model.wv.syn0 # each row is a word vector, sorted by word frequency

targetWordsIndex = [model.wv.vocab[w].index for w in targetWords]

# compute the average similarity of two set of vectors
def average_sim(A, B):
    ans = 0
    for a, b in zip(A, B):
        aNorm = np.linalg.norm(a)
        bNorm = np.linalg.norm(b)
        ans += (np.inner(a, b) / (aNorm * bNorm)) * 4/(2 + aNorm / bNorm + bNorm / aNorm)
    return ans/A.shape[0]

# compute the average distance of two set of vectors
def average_distance(A, B):
    ans = 0
    for a, b in zip(A, B):
        ans += np.linalg.norm(a - b)
    ans /= A.shape[0]
    return ans

# select the word vectors and context vectors of the target words (some numpy tricks)
N = len(targetWords)
targetWordVec = wordVectors[targetWordsIndex, :]
targetContextVec = contextVectors[targetWordsIndex, :]

# wordVectors = targetWordVec
# contextVectors = targetContextVec
# targetContextVec = np.random.random(size=targetWordVec.shape)

print(f'target words average similarity before alignment: {average_sim(targetWordVec, targetContextVec)}')
# topKWordsIndexExcludeTarget = [i for i in range(min(K, len(model.wv.index_to_key))) if model.wv.index_to_key[i] not in targetWords]

def tSNE_not_align(targetWordsIndex, wordVectors, contextVectors, tSNE_Size, onlyShowTarget = True):
    targetWordVec = wordVectors[targetWordsIndex, :]
    targetContextVec = contextVectors[targetWordsIndex, :]
    topKWordVec = wordVectors[:tSNE_Size]
    topKContextVec = contextVectors[:tSNE_Size]
    avgSim = average_sim(targetWordVec, targetContextVec)
    avgDistance = average_distance(targetWordVec, targetContextVec)
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
    plt.title(f'Unaligned word and context vectors for target words. \nAvg.similarity = {avgSim:.2f} \nAvg.distance = {avgDistance}.')
    plt.tight_layout()
    plt.show()

# tSNE_not_align(targetWordsIndex, wordVectors, contextVectors, K, True)

# ignore this one
def tSNE_align_by_target_words(wordVectors, contextVectors):
    plt.figure(figsize=GraphSize)
    for i, N in enumerate([10, 100, 1000, 10000]):
        targetWordsIndex = random.sample(range(wordVectors.shape[0]), N)
        targetWordVec = wordVectors[targetWordsIndex, :]
        targetContextVec = contextVectors[targetWordsIndex, :]
        avgSimBeforeAlignment = average_sim(targetWordVec, targetContextVec)
        avgDistBeforeAlignment = average_distance(targetWordVec, targetContextVec)
        contextToWord = absolute_orientation(targetContextVec, targetWordVec, targetContextVec)
        avgSimAfterAlignment = average_sim(targetWordVec, contextToWord)
        avgDistAfterAlignment = average_distance(targetWordVec, contextToWord)
        logging.info(f"tSNE_align_by_target_words: Performing t-SNE for N = {N}.")
        points = reduce_dimensions(np.concatenate((targetWordVec, contextToWord)))
        X1 = [v[0] for v in points[:N]]
        Y1 = [v[1] for v in points[:N]]
        X2 = [v[0] for v in points[N:]]
        Y2 = [v[1] for v in points[N:]]
        plt.subplot(2, 2, i + 1)
        plt.scatter(X1, Y1, c='blue', alpha=0.3, s=5)
        plt.scatter(X2, Y2, c='red', alpha=0.3, s=5)
        plt.title(f'{N=}. \nAvg.similarity: {avgSimBeforeAlignment:.2f} --> {avgSimAfterAlignment:.2f} \nAvg.similarity: {avgDistBeforeAlignment:.2f} --> {avgDistAfterAlignment:.2f}')
    plt.tight_layout()
    plt.show()

# tSNE_align_by_target_words(wordVectors, contextVectors)

# You only need to read this function.
# align the target words by random words, i.e., learn the transformation from random words
def tSNE_align_by_random_words(targetWordsIndex, wordVectors, contextVectors, showAlignWords = False, tSNEtargetOnly = False):
    plt.figure(figsize=GraphSize)
    K = len(targetWordsIndex) # the number of target words
    referWordsIndex = targetWordsIndex
    while len(set(targetWordsIndex) & set(referWordsIndex)) > 0:
        referWordsIndex = random.sample(range(wordVectors.shape[0]), K)

    for i, N in enumerate([500, 1000, 2000, 4000]):
        alignWordsIndex = targetWordsIndex
        while len(set(alignWordsIndex) & set(targetWordsIndex)) > 0 or len(set(alignWordsIndex) & set(referWordsIndex)) > 0:
            alignWordsIndex = random.sample(range(wordVectors.shape[0]), N)
        targetWordVec = wordVectors[targetWordsIndex, :]
        targetContextVec = contextVectors[targetWordsIndex, :]
        referWordVec = wordVectors[referWordsIndex, :]
        referContextVec = contextVectors[referWordsIndex, :]
        
        alignWordVec = wordVectors[alignWordsIndex, :] # word vectors for alignment
        alignContextVec = contextVectors[alignWordsIndex, :] # context vectors for alignment

        targetSimBeforeAlignment = average_sim(targetWordVec, targetContextVec)
        targetDistBeforeAlignment = average_distance(targetWordVec, targetContextVec)
        referSimBeforeAlignment = average_sim(targetWordVec, referContextVec)
        referDistBeforeAlignment = average_distance(targetWordVec, referContextVec)

        alignContextVecToWord = absolute_orientation(alignContextVec, alignWordVec, alignContextVec)
        targetContextVecToWord = absolute_orientation(alignContextVec, alignWordVec, targetContextVec)
        referContextVecToWord = absolute_orientation(alignContextVec, alignWordVec, referContextVec)

        targetSimAfterAlignment = average_sim(targetWordVec, targetContextVecToWord)
        targetDistAfterAlignment = average_distance(targetWordVec, targetContextVecToWord)
        referSimAfterAlignment = average_sim(targetWordVec, referContextVecToWord)
        referDistAfterAlignment = average_distance(targetWordVec, referContextVecToWord)

        plt.subplot(2, 2, i + 1)
        if tSNEtargetOnly:
            points = reduce_dimensions(np.concatenate((targetWordVec, targetContextVecToWord)))
            targetWordPoints = points[:K]
            targetContextPoints = points[K:]
        else:
            logging.info(f"tSNE_align_by_random_words: Performing t-SNE for N = {N}.")
            points = reduce_dimensions(np.concatenate((alignWordVec, alignContextVecToWord, targetWordVec, targetContextVecToWord)))
            alignWordPoints = points[:N]
            alignContextPoints = points[N:2*N]
            targetWordPoints = points[2*N:2*N+K]
            targetContextPoints = points[2*N+K:2*N+2*K]
            # referContextPoints = points[2*N+2*K:]
            if showAlignWords:
                alignWordX = [v[0] for v in alignWordPoints]
                alignWordY = [v[1] for v in alignWordPoints]
                alignContextX = [v[0] for v in alignContextPoints]
                alignContextY = [v[1] for v in alignContextPoints]
                
                plt.scatter(alignWordX, alignWordY, c='blue', alpha=0.3, s=5)
                plt.scatter(alignContextX, alignContextY, c='red', alpha=0.3, s=5)
            
        targetWordX = [v[0] for v in targetWordPoints]
        targetWordY = [v[1] for v in targetWordPoints]
        targetcontextX = [v[0] for v in targetContextPoints]
        targetcontextY = [v[1] for v in targetContextPoints]

        plt.scatter(targetWordX, targetWordY, c='blue', marker='*')
        plt.scatter(targetcontextX, targetcontextY, c='red', marker='*')
        for i in range(len(targetWords)):
            plt.annotate(targetWords[i] + '_word', (targetWordX[i], targetWordY[i]))
            plt.annotate(targetWords[i] + '_context', (targetcontextX[i], targetcontextY[i]))
        plt.title(f'{N=}. \nTarget similarity: {targetSimBeforeAlignment:.2f} --> {targetSimAfterAlignment:.2f}\nReference similarity: {referSimBeforeAlignment:.2f} --> {referSimAfterAlignment:.2f} \nTarget distance: {targetDistBeforeAlignment:.2f} --> {targetDistAfterAlignment:.2f}\nReference distance: {referDistBeforeAlignment:.2f} --> {referDistAfterAlignment:.2f}')
    plt.tight_layout()
    plt.show()

tSNE_align_by_random_words(targetWordsIndex, wordVectors, contextVectors, showAlignWords = True, tSNEtargetOnly = False)

def tSNE_align_random_vectors():
    plt.figure(figsize=GraphSize)
    for i, N in enumerate([500, 1000, 2000, 4000]):
        fromPoints = (np.random.random(size=(N, 400)) - 0.5) * 2
        toPoints = (np.random.random(size=(N, 400)) - 0.5) * 2

        avgSimBeforeAlignment = average_sim(fromPoints, toPoints)
        avgDistBeforeAlignment = average_distance(fromPoints, toPoints)
        alignedPoints = absolute_orientation(fromPoints, toPoints, fromPoints)
        avgSimAfterAlignment = average_sim(toPoints, alignedPoints)
        avgDistAfterAlignment = average_distance(toPoints, alignedPoints)
        logging.info(f"tSNE_align_random_vectors: Performing t-SNE for N = {N}.")
        points = reduce_dimensions(np.concatenate((toPoints, alignedPoints)))
        plt.subplot(2, 2, i + 1)

        X1 = [v[0] for v in points[:points.shape[0]//2]]
        Y1 = [v[1] for v in points[:points.shape[0]//2]]
        X2 = [v[0] for v in points[points.shape[0]//2:]]
        Y2 = [v[1] for v in points[points.shape[0]//2:]]
        
        plt.scatter(X1, Y1, c='blue', alpha=0.3, s=5)
        plt.scatter(X2, Y2, c='red', alpha=0.3, s=5)
        
        plt.title(f'{N=}. \nAvg.similarity: {avgSimBeforeAlignment:.2f} --> {avgSimAfterAlignment:.2f} \nAvg.distance: {avgDistBeforeAlignment:.2f} --> {avgDistAfterAlignment:.2f}')
    plt.tight_layout()
    plt.show()

def tSNE_align_random_vectors_by_random_vectors(showAlignPoints = False):
    plt.figure(figsize=GraphSize)
    K = 10
    for i, N in enumerate([500, 1000, 2000, 4000]):
        fromPointsTrain = (np.random.random(size=(N, 400)) - .5) * 2
        toPointsTrain = (np.random.random(size=(N, 400)) - .5) * 2
        fromPointsTest = (np.random.random(size=(K, 400)) - .5) * 2
        toPointsTest = (np.random.random(size=(K, 400)) - .5) * 2


        avgSimBeforeAlignment = average_sim(fromPointsTest, toPointsTest)
        avgDistBeforeAlignment = average_distance(fromPointsTest, toPointsTest)
        alignedPoints = absolute_orientation(fromPointsTrain, toPointsTrain, np.concatenate((fromPointsTest, fromPointsTrain)))
        avgSimAfterAlignment = average_sim(toPointsTest, alignedPoints[:K])
        avgDistAfterAlignment = average_distance(toPointsTest, alignedPoints[:K])
        logging.info(f"tSNE_align_random_vectors: Performing t-SNE for N = {N}.")
        points = reduce_dimensions(np.concatenate((alignedPoints, toPointsTrain, toPointsTest)))
        plt.subplot(2, 2, i + 1)
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
        plt.title(f'{N=}. \nAvg.similarity: {avgSimBeforeAlignment:.2f} --> {avgSimAfterAlignment:.2f} \nAvg.distance: {avgDistBeforeAlignment:.2f} --> {avgDistAfterAlignment:.2f}')
    plt.tight_layout()
    plt.show()

# tSNE_align_random_vectors_by_random_vectors()
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