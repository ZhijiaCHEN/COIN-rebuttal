import gensim.downloader as api
from gensim.test.utils import datapath
from gensim import utils
import gensim.models
import logging
import numpy
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MyCorpus:
    """An iterator that yields sentences (lists of str)."""
    def __iter__(self):
        corpus_path = datapath('lee_background.cor')
        for line in open(corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)
import json
info = api.info()
print(json.dumps(info, indent=4))

modelFull = 'wiki-english-20171001'
modelDebug = 'semeval-2016-2017-task3-subtaskA-unannotated'
modelName = modelDebug
sentences = api.load(modelName)
model = gensim.models.Word2Vec(sentences=sentences, vector_size=300, sorted_vocab=1, max_final_vocab=None, max_vocab_size=None)
model.save(f'{modelName}.model')
wordVectors = model.wv.vectors
if hasattr(model, 'syn1'):
    contextVectors = model.syn1
else:
    contextVectors = model.syn1neg
numpy.save('wordVectors', wordVectors)
numpy.save('contextVectors', contextVectors)
