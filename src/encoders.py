from abc import ABC, abstractmethod
import tensorflow as tf 
import tensorflow_hub as hub
import tensorflow_text
import numpy as np
<<<<<<< HEAD
=======
import simpleneighbors
>>>>>>> 3ad38caaa78c49d4f4cf7414a3e48f5c305dc3de
import logging


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
<<<<<<< HEAD
        embs = self.module.signatures['question_encoder'](
            tf.constant(queries))
        return embs
=======
        all_vectors = []

        for i in range(0, len(queries), self.BATCH_SIZE):
            batch = queries[i:i+self.BATCH_SIZE]
            embs = self.module.signatures['question_encoder'](
                tf.constant(batch))
            all_vectors.append(embs)
        return np.array(all_vectors).astype(np.float32)
>>>>>>> 3ad38caaa78c49d4f4cf7414a3e48f5c305dc3de


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
<<<<<<< HEAD
        embs = self.module.signatures['response_encoder'](
            input=tf.constant(responses), context=tf.constant(contexts))
        return embs
=======
        all_vectors = []
        
        for i in range(0, len(responses), self.BATCH_SIZE):
            batch = responses[i:i+self.BATCH_SIZE]
            embs = self.module.signatures['response_encoder'](
                input=tf.constant(responses), context=tf.constant(contexts))
            all_vectors.append(embs)
        return np.array(all_vectors).astype(np.float32)

    def save_vector_index(self, responses, response_embeddings, index_prefix, metric='angular'):
        """
        builds precomputed vector index for faster serving. uses the Annoy library by default. 
        :param emb_dim_size: 
        :param metric:
        :param response_emb_tup:
        :param index_prefix: 
        :returns simpleneighbors index for nearest neighbors vector lookup
        """
        index = simpleneighbors.SimpleNeighbors(len(response_embeddings['output'][0]), metric)
        response_emb_tup = list(zip(responses, response_embeddings))
        index.feed(response_emb_tup)
        index.build()
        print('index created and built')
        index.save(index_prefix)
        print('index saved')
>>>>>>> 3ad38caaa78c49d4f4cf7414a3e48f5c305dc3de
