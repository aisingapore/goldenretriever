from abc import ABC, abstractmethod
import tensorflow as tf 
import tensorflow_hub as hub
import tensorflow_text
import numpy as np
import simpleneighbors
import logging
import pickle


logger = logging .getLogger(__name__)


class Encoder(ABC):
    """a shared encoder interface
    Each encoder should provide an encode() method and a FEATURE_SZE
    constant indicating the size of the encoded vectors"""

    FEATURE_SIZE = int 

    @abstractmethod
    def encode(self, data):
        pass


class USEQueryEncoder(Encoder):
    """query encoder using Google USE QA model"""

    FEATURE_SIZE = 512
    BATCH_SIZE = 32

    def __init__(self, tfhub_module_url):
        self.module = hub.load(tfhub_module_url)

    def encode(self, queries):
        """
        Encode queries into vectors with 512 dims
        :param queries: iterable of query strings 
        :returns: np.array of vectorized queries 
        """
        logger.debug(f"encoding queries")
        embs = self.module.signatures['question_encoder'](
            tf.constant(queries))
        return embs


class USEResponseEncoder(ABC):
    """response encoder using GOOGLE USE QA model"""
    FEATURE_SIZE = 512
    BATCH_SIZE = 32

    def __init__(self, tfhub_module_url):
        self.module = hub.load(tfhub_module_url)

    def encode(self, responses, contexts):
        """
        Encode responses into vectors with 512 dims 
        :param responses: iterable of response strings
        :param contexts: iterable of context strings. if no context strings are available, provide
                            iterable of empty strings 
        :returns:  np.array of vectorized responses 
        """
        logger.debug(f"encoding responses")
        embs = self.module.signatures['response_encoder'](
            input=tf.constant(responses), context=tf.constant(contexts))
        return embs


class SimpleNNIndex():

    def __init__(self, emb_dim_size, metric='angular'):
        self.nn_index = simpleneighbors.SimpleNeighbors(emb_dim_size, metric)

    def build(self, sentence_strings, sentence_embeddings):
        """
        builds precomputed vector index for faster serving. uses the Annoy library by default. 
        :param emb_dim_size: 
        :param response_emb_tup:
        :param index_prefix: 
        :returns simpleneighbors index for nearest neighbors vector lookup
        """
        sentence_emb_tup = list(zip(sentence_strings, sentence_embeddings['outputs'].numpy()))
        self.nn_index.feed(sentence_emb_tup)
        self.nn_index.build()
        print('index built')
    
    def query(self, query_embeddings, num_nbrs):
        return self.nn_index.nearest(query_embeddings, num_nbrs)

    def save(self, index_prefix):
        self.nn_index.save(index_prefix)
        print('index saved')

    @classmethod
    def load(cls, prefix):
        with open(prefix + '-data.pkl', 'rb') as fh:
            data = pickle.load(fh)
        new_index = cls(emb_dim_size=data['dims'], metric=data['metric'])
        new_index.id_map = data['id_map']
        new_index.corpus = data['corpus']
        new_index.i = data['i']
        new_index.built = data['built']
        new_index.backend.load(prefix + ".idx")        
        return new_index


if __name__=='__main__':
    import pandas as pd 
    nrf = pd.read_csv('data/nrf.csv')
    nrf.head()

    queries = nrf['query_str'].fillna('na').tolist()[0:5]
    responses = nrf['ans_str'].tolist()[0:5]
    contexts = [''] * len(responses)
    tfhub_module_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3'

    query_embeddings = USEQueryEncoder(tfhub_module_url).encode(queries)
    response_embeddings = USEResponseEncoder(tfhub_module_url).encode(responses, contexts)
    print(f"response as {len(response_embeddings['outputs'][0])}")
    
    simple_nn = SimpleNNIndex(len(response_embeddings['outputs'][0]))
    simple_nn.build(responses, response_embeddings)
    simple_nn.query(query_embeddings['outputs'][0], 1)
    simple_nn.save('nrf')
    
    saved_index = SimpleNNIndex.load('nrf')
    print('loaded index')
    print(saved_index.query(query_embeddings['outputs'][0], 1))
