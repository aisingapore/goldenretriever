from abc import ABC, abstractmethod
import tensorflow as tf 
import tensorflow_hub as hub
import tensorflow_text
import numpy as np
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
