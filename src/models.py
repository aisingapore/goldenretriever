
import numpy as np
import datetime

from abc import ABC, abstractmethod
from sklearn.metrics.pairwise import cosine_similarity

from .data_handler.data_preprocessing import clean_txt
from .data_handler.kb_handler import kb


class Model(ABC):
    """
    a shared model interface where
    each model should provide
    finetune, predict, make_query, export_encoder,
    restore_encoder methods
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def finetune(self):
        """
        finetunes encoder
        """
        pass

    @abstractmethod
    def predict(self):
        """
        encode method of encoder
        will be used to vectorize texts
        """
        pass

    @abstractmethod
    def load_kb(self):
        """
        load and encode knowledge bases to return predictions
        """

    @abstractmethod
    def make_query(self):
        """
        uses predict method to vectorize texts
        and provides relevant responses based on
        given specifications (eg. num responses) to user
        """
        pass

    @abstractmethod
    def export_encoder(self):
        """
        export finetuned weights
        """
        pass

    @abstractmethod
    def restore_encoder(self):
        """
        restores encoder with finetuned weights
        """
        pass


class GoldenRetriever(Model):
    def __init__(self, encoder):
        self.encoder = encoder

    def finetune(self, question, answer, margin=0.3,
                 loss='triplet', context=[], neg_answer=[],
                 neg_answer_context=[], label=[]):

        cost_value = self.encoder.finetune_weights(question, answer,
                                                   margin=margin, loss=loss,
                                                   context=context,
                                                   neg_answer=neg_answer,
                                                   neg_answer_context=neg_answer_context,
                                                   label=label)
        return cost_value

    def predict(self, text, context=None, string_type='response'):
        encoded_responses = self.encoder.encode(text, context=context,
                                                string_type=string_type)

        return encoded_responses

    def load_kb(self, kb_):
        """
        Load the knowledge base or bases
        :param kb: kb object as defined in kb_handler
        """

        self.kb = {}

        if type(kb_) == kb:
            context_and_raw_string = kb_.responses.context_string.fillna('') + \
                ' ' + kb_.responses.raw_string.fillna('')

            kb_.vectorised_responses = self.predict(clean_txt(context_and_raw_string),
                                                    string_type='response')

            self.kb[kb_.name] = kb_
            print(f'{datetime.datetime.now()} : kb loaded - {kb_.name} ')

        elif hasattr(kb_, '__iter__'):
            for one_kb in kb_:
                self.load_kb(one_kb)

    def make_query(self, querystring, top_k=5, index=False,
                   predict_type='query', kb_name='default_kb'):
        """
        Make a query against the stored vectorized knowledge.

        :type type: str
        :type kb_name: str
        :type index: boolean
        :param type: can be 'query' or 'response'. Use to compare statements
        :param type: the name of knowledge base in the knowledge dictionary
        :param type: Choose index=True to return sorted index of matches. 
        :return: Top K vectorized answers and their scores
        """

        similarity_score = cosine_similarity(self.kb[kb_name].vectorised_responses,
                                             self.predict([querystring],
                                             string_type=predict_type))

        sortargs = np.flip(similarity_score.argsort(axis=0))
        sortargs = [x[0] for x in sortargs]

        # sorted answer conditional if there is a context string, 
        # then include as a line-separated pre-text
        sorted_ans = []
        for i in sortargs:
            ans = self.kb[kb_name].responses.context_string.iloc[i] + \
                    '\n' + self.kb[kb_name].responses.raw_string.iloc[i] \
                    if self.kb[kb_name].responses.context_string.iloc[i] != '' \
                    else self.kb[kb_name].responses.raw_string.iloc[i]

            sorted_ans.append(ans)

        if index:
            return sorted_ans[:top_k], sortargs[:top_k]

        return sorted_ans[:top_k], similarity_score[sortargs[:top_k]] 

    def export_encoder(self, save_dir=None):
        '''
        Path should include partial filename.
        https://www.tensorflow.org/api_docs/python/tf/saved_model/save
        '''
        self.encoder.save_weights(save_dir=save_dir)

    def restore_encoder(self, save_dir):
        """
        Signatures need to be re-init after weights are loaded.
        """
        self.encoder.restore_weights(save_dir=save_dir)
