import logging
import os
import tensorflow as tf 
# import tensorflow_addons as tfa
import tensorflow_hub as hub

from abc import ABC, abstractmethod
from transformers import AlbertTokenizer, TFAlbertModel

from .loss_functions import triplet_loss
from .tokenizers.bert_tokenization import FullTokenizer, preprocess_str


logger = logging .getLogger(__name__)


class Encoder(ABC):
    """a shared encoder interface
    Each encoder should provide an encode() method"""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def encode(self):
        pass

    @abstractmethod
    def finetune_weights(self):
        pass

    @abstractmethod
    def save_weights(self):
        pass

    @abstractmethod
    def restore_weights(self):
        pass


class USEEncoder(Encoder):
    def __init__(self, **kwargs):

        # variables to be finetuned
        # self.v=['QA/Final/Response_tuning/ResidualHidden_1/dense/kernel','QA/Final/Response_tuning/ResidualHidden_0/dense/kernel', 'QA/Final/Response_tuning/ResidualHidden_1/AdjustDepth/projection/kernel']
        self.v = ['QA/Final/Response_tuning/ResidualHidden_1/AdjustDepth/projection/kernel']

        if kwargs:
            self.opt_params = kwargs
        else:
            # good defaults for params
            self.opt_params = {
                'learning_rate': 0.001,
                'beta_1': 0.9,
                'beta_2': 0.999,
                'epsilon': 1e-07
            }

        # init saved model
        self.embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder-qa/3')
        self.init_signatures()

    def init_signatures(self):

        # re-initialize the references to the model signatures
        self.question_encoder = self.embed.signatures['question_encoder']
        self.response_encoder = self.embed.signatures['response_encoder']
        self.neg_response_encoder = self.embed.signatures['response_encoder']
        print('model initiated!')

        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(**self.opt_params)

        # retrieve the weights we want to finetune.
        self.var_finetune = [x for x in self.embed.variables
                             for vv in self.v if vv in x.name]

    def encode(self, text, context=None, string_type=None):

        if string_type == 'query':
            if isinstance(text, str):
                return self.question_encoder(tf.constant([text]))['outputs']

            elif hasattr(text, '__iter__'):
                return tf.concat(
                    [self.question_encoder(tf.constant([one_text]))['outputs']
                     for one_text in text], axis=0
                )

        elif string_type == 'response':
            """
            A frequent error is OOM - Error recorded below.
            The fix is to encode each entry separately.
            This is implemented in a list comprehension.
            """
            if not context:
                context = text

            if isinstance(text, str):
                return self.response_encoder(
                    input=tf.constant([text]),
                    context=tf.constant([context])
                )['outputs']

            elif hasattr(text, '__iter__'):
                encoded_responses = [self.response_encoder(input=tf.constant([t]),
                                     context=tf.constant([c]))['outputs']
                                     for t, c in zip(text, context)]

                encoded_responses_tensor = tf.concat(encoded_responses, axis=0)
                return encoded_responses_tensor

        else:
            print('Type of prediction not defined')

    def finetune_weights(self, question, answer, margin=0.3,
                         loss='triplet', context=[], neg_answer=[],
                         neg_answer_context=[], label=[]):
        """
        Finetune the model

        Parameters:
        loss(string): loss function can be 'triplet', 'cosine' and 'contrastive'

        """
        self.cost_history = []
        with tf.GradientTape() as tape:
            # get encodings
            question_embeddings = self.question_encoder(
                tf.constant(question)
            )['outputs']

            response_embeddings = self.response_encoder(
                input=tf.constant(answer),
                context=tf.constant(context)
            )['outputs']

            if loss == 'cosine':
                """
                # https://www.tensorflow.org/api_docs/python/tf/keras/losses/CosineSimilarity

                """
                self.cost = tf.keras.losses.CosineSimilarity(axis=1)
                cost_value = self.cost(question_embeddings,
                                       response_embeddings)

            elif loss == 'contrastive':
                """
                https://www.tensorflow.org/addons/api_docs/python/tfa/losses/ContrastiveLoss

                y_true to be a vector of binary labels
                y_hat to be the respective distances

                """
                self.cosine_dist = tf.keras.losses.CosineSimilarity(axis=1)
                cosine_dist_value = self.cosine_dist(question_embeddings,
                                                     response_embeddings)

                self.cost = tfa.losses.contrastive.ContrastiveLoss(margin=margin)
                cost_value = self.cost(label, cosine_dist_value)

            elif loss == 'triplet':
                """
                Triplet loss uses a non-official self-implementated loss function outside of TF based on cosine distance

                """
                neg_response_embeddings = self.neg_response_encoder(
                    input=tf.constant(neg_answer),
                    context=tf.constant(neg_answer_context)
                )['outputs']

                cost_value = triplet_loss(
                    question_embeddings,
                    response_embeddings,
                    neg_response_embeddings,
                    margin=margin
                )

        # record loss
        self.cost_history.append(cost_value.numpy().mean())

        # apply gradient
        grads = tape.gradient(cost_value, self.var_finetune)
        self.optimizer.apply_gradients(zip(grads, self.var_finetune))

        return cost_value.numpy().mean()

    def save_weights(self, save_dir=None):
        '''
        Path should include partial filename.
        https://www.tensorflow.org/api_docs/python/tf/saved_model/save

        '''
        tf.saved_model.save(
            self.embed,
            save_dir,
            signatures={
                'default': self.embed.signatures['default'],
                'response_encoder':self.embed.signatures['response_encoder'],
                'question_encoder':self.embed.signatures['question_encoder']
            }
        )

    def restore_weights(self, save_dir=None):
        """
        Signatures need to be re-init after weights are loaded.

        """
        self.embed = tf.saved_model.load(save_dir)
        self.init_signatures()


class ALBERTEncoder(Encoder):

    def __init__(self, max_seq_length=512):

        # ALBERT unique params
        self.max_seq_length = max_seq_length

        # GR params
        self.vectorized_knowledge = {}
        self.text = {}
        self.questions = {}
        self.opt_params = {
            'learning_rate': 0.001,
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-07
        }

        # init saved model
        self.albert_layer = TFAlbertModel.from_pretrained('albert-base-v2')

        # writing the model for the training tasks
        # get inputs

        res_id = tf.keras.layers.Input(
            shape=(self.max_seq_length,),
            name="input_ids",
            dtype='int32'
        )

        res_mask = tf.keras.layers.Input(
            shape=(self.max_seq_length,),
            name="input_masks",
            dtype='int32'
        )

        res_segment = tf.keras.layers.Input(
            shape=(self.max_seq_length,),
            name="input_seg",
            dtype='int32'
        )

        # encode the three inputs
        _, res_pooled = self.albert_layer([res_id, res_mask, res_segment])

        # dense layer specifically for
        self.response_encoder = tf.keras.layers.Dense(
            768, input_shape=(768,),
            name='response_dense_layer'
        )

        encoded_response = self.response_encoder(res_pooled)

        # init model
        self.albert_model = tf.keras.Model(
            inputs=[res_id, res_mask, res_segment],
            outputs=encoded_response
        )

        print("Initializing tokenizer and optimizer")
        self.init_signatures()

    def init_signatures(self):
        """
        Re-init references to layers and model attributes        
        When restoring the model, the references to the vocab file / layers would be lost.
        """

        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

        # init optimizer
        self.optimizer = tf.keras.optimizers.Adam(**self.opt_params)
        self.cost_history = []

        # TF-Hub page recommentds finetuning all weights
        # "All parameters in the module are trainable,
        # and fine-tuning all parameters is the recommended practice."
        self.var_finetune = self.albert_model.variables

        print('model initiated!')

    def _encode_one_str(self, text, string_type='response'):
        """
        Return the tensor representing embedding of input text.
        Type can be 'query' or 'response' 

        args:
            text: (str or iterable of str) This contains the text that is required to be encoded
            type: (str) Either 'response' or 'query'. Default is 'response'. 
                        This tells GR to either use the response or query encoder
                        but in the case of BERT, this argument is ignored

        Return:
            pooled_embedding: (tf.tensor) contains the 768 dim encoding of the input text
        """

        if string_type == 'query':
            question_id_mask_seg = preprocess_str(
                text, self.max_seq_length,
                self.tokenizer
            )

            question_embedding = self.albert_layer(
                [tf.constant(question_id_mask_seg[0]),
                 tf.constant(question_id_mask_seg[1]),
                 tf.constant(question_id_mask_seg[2])]
            )[1]

            return question_embedding

        if string_type == 'response':
            response_id_mask_seg = preprocess_str(
                text, self.max_seq_length,
                self.tokenizer
            )

            response_embedding = self.albert_model(
                [tf.constant(response_id_mask_seg[0]),
                 tf.constant(response_id_mask_seg[1]),
                 tf.constant(response_id_mask_seg[2])]
            )

            return response_embedding

    def encode(self, text, context=None, string_type='response'):
        encoded_strings = [self._encode_one_str(t, string_type=string_type)
                           for t in text]

        encoded_tensor = tf.concat(encoded_strings, axis=0)
        return encoded_tensor

    def finetune_weights(self, question, answer, margin=0.3,
                         loss='triplet', context=[], neg_answer=[],
                         neg_answer_context=[], label=[]):
        """
        Finetune model with GradientTape.

        args:
            question: (str or iterable of str) This contains the questions that is required to be encoded
            answer: (str or iterable of str) This contains the response that is required to be encoded
            margin: (float) margin tuning parameter for triplet / contrastive loss
            loss: (str) name of loss function. ('cosine', 'contrastive', 'triplet'). Default setting is 'triplet.
            context: (str or iterable of str) Ignored for BERT/ALBERT
            neg_answer: (str or iterable of str) This contains the distractor responses that is required to be encoded
            neg_answer_context: (str or iterable of str) Ignored for BERT/ALBERT
            label: (list of int) This contain the label for contrastive loss

        return:
            loss_value: (float) This returns the loss of the training task

        """
        question_id_mask_seg = preprocess_str(
            question, self.max_seq_length,
            self.tokenizer
        )
        response_id_mask_seg = preprocess_str(
            answer, self.max_seq_length,
            self.tokenizer
        )

        # for eager execution finetuning
        with tf.GradientTape() as tape:

            # tf-hub's keras layer can take the lists directly
            # but the bert_model object needs the inputs to be tf.constants
            question_embeddings = self.albert_layer(
                [tf.constant(question_id_mask_seg[0]),
                 tf.constant(question_id_mask_seg[1]),
                 tf.constant(question_id_mask_seg[2])]
            )[1]

            response_embeddings = self.albert_model(
                [tf.constant(response_id_mask_seg[0]),
                 tf.constant(response_id_mask_seg[1]),
                 tf.constant(response_id_mask_seg[2])]
            )

            if loss == 'cosine':
                self.cost = tf.keras.losses.CosineSimilarity(axis=1)
                cost_value = self.cost(question_embeddings,
                                       response_embeddings)

            elif loss == 'contrastive':
                """
                https://www.tensorflow.org/addons/api_docs/python/tfa/losses/ContrastiveLoss

                y_true to be a vector of binary labels
                y_hat to be the respective distances
                """
                self.cosine_dist = tf.keras.losses.CosineSimilarity(axis=1)
                cosine_dist_value = self.cosine_dist(question_embeddings,
                                                     response_embeddings)

                self.cost = tfa.losses.contrastive.ContrastiveLoss(margin=margin)
                cost_value = self.cost(label, cosine_dist_value)

            elif loss == 'triplet':
                """
                Triplet loss uses a non-official self-implementated loss function outside of TF based on cosine distance
                """
                # encode the negative response
                neg_answer_id_mask_seg = preprocess_str(
                    neg_answer,
                    self.max_seq_length,
                    self.tokenizer
                )

                neg_response_embeddings = self.albert_model(
                    [tf.constant(neg_answer_id_mask_seg[0]),
                     tf.constant(neg_answer_id_mask_seg[1]),
                     tf.constant(neg_answer_id_mask_seg[2])]
                )

                cost_value = triplet_loss(question_embeddings,
                                          response_embeddings,
                                          neg_response_embeddings)

            # record loss
            self.cost_history.append(cost_value.numpy().mean())

        # apply gradient
        self.grads = tape.gradient(cost_value, self.var_finetune)
        self.optimizer.apply_gradients(zip(self.grads, self.var_finetune))

        return cost_value.numpy().mean()

    def save_weights(self, save_dir=None):
        '''
        Save the BERT model into a directory

        The original saving procedure taken from:
        https://github.com/tensorflow/models/blob/master/official/nlp/bert/export_tfhub.py
        The tf-hub module does includes the str vocab_file directory and do_lower_case boolean
        The future restore() function should depend on a fresh copy of the vocab file,
        because loading the model in a different directory demands a different directory for the vocab.
        However, the string vocab_file directory and do_lower_case boolean is kept to the saved model anyway
        '''
        # model.save does not work if there are layers that are subclassed (eg. huggingface models)
        save_path = os.path.join(save_dir, 'model')
        self.albert_model.save_weights(save_path)

    def restore_weights(self, save_dir=None):
        """
        Load weights from savepath

        Args:
            savepath: (str) dir path of the weights
        """
        save_path = os.path.join(save_dir, 'model')
        self.albert_model.load_weights(save_path)


class BERTEncoder(Encoder):
    def __init__(self, max_seq_length=512):

        # BERT unique params
        self.max_seq_length = max_seq_length

        # GR params
        self.vectorized_knowledge = {}
        self.text = {}
        self.questions = {}
        self.opt_params = {
            'learning_rate': 0.001,
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-07
        }

        # init saved model
        # self.bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=True)  # uncased and smaller model
        self.bert_layer = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
            trainable=True
        )

        self.vocab_file = self.bert_layer.resolved_object.vocab_file.asset_path.numpy()
        self.do_lower_case = self.bert_layer.resolved_object.do_lower_case.numpy()

        # writing the model for the training tasks
        # get inputs
        res_id = tf.keras.layers.Input(
            shape=(self.max_seq_length,),
            name="input_ids",
            dtype='int32'
        )

        res_mask = tf.keras.layers.Input(
            shape=(self.max_seq_length,),
            name="input_masks",
            dtype='int32'
        )

        res_segment = tf.keras.layers.Input(
            shape=(self.max_seq_length,),
            name="input_seg",dtype='int32'
        )

        # encode the three inputs
        res_pooled, res_seq = self.bert_layer([res_id, res_mask, res_segment])

        # dense layer specifically for
        self.response_encoder = tf.keras.layers.Dense(
            768, input_shape=(768,),
            name='response_dense_layer'
        )

        encoded_response = self.response_encoder(res_pooled)

        # init model
        self.bert_model = tf.keras.Model(
            inputs=[res_id, res_mask, res_segment],
            outputs=encoded_response
        )

        print("Downloaded model from Hub, initializing tokenizer and optimizer")
        self.init_signatures()

    def init_signatures(self):
        """
        Re-init references to layers and model attributes        
        When restoring the model, the references to the vocab file / layers would be lost.
        """
        # init tokenizer from hub layer
        self.tokenizer = FullTokenizer(self.vocab_file, self.do_lower_case)

        # init optimizer
        self.optimizer = tf.keras.optimizers.Adam(**self.opt_params)
        self.cost_history = []

        # bert layer name
        self.bert_layer_name = [layer.name for layer in self.bert_model.layers
                                if layer.name.startswith('keras_layer')][0]

        # TF-Hub page recommentds finetuning all weights
        # "All parameters in the module are trainable,
        # and fine-tuning all parameters is the recommended practice."
        self.var_finetune = self.bert_model.variables

        print('model initiated!')

    def encode(self, text, context=None, string_type='response'):
        """
        Return the tensor representing embedding of input text.
        Type can be 'query' or 'response' 

        args:
            text: (str or iterable of str) This contains the text that is required to be encoded
            type: (str) Either 'response' or 'query'. Default is 'response'. 
                        This tells GR to either use the response or query encoder
                        but in the case of BERT, this argument is ignored

        Return:
            pooled_embedding: (tf.tensor) contains the 768 dim encoding of the input text
        """

        if string_type == 'query':
            question_id_mask_seg = preprocess_str(
                text, self.max_seq_length,
                self.tokenizer
            )

            question_embeddings, q_sequence_output = self.bert_model.get_layer(self.bert_layer_name)(question_id_mask_seg)
            return question_embeddings

        elif string_type == 'response':
            response_id_mask_seg = preprocess_str(
                text, self.max_seq_length,
                self.tokenizer
            )

            response_embeddings = self.bert_model(
                [tf.constant(response_id_mask_seg[0]),
                 tf.constant(response_id_mask_seg[1]),
                 tf.constant(response_id_mask_seg[2])]
            )

            return response_embeddings

    def finetune_weights(self, question, answer, margin=0.3,
                         loss='triplet', context=[], neg_answer=[],
                         neg_answer_context=[], label=[]):
        """
        Finetune model with GradientTape.

        args:
            question: (str or iterable of str) This contains the questions that is required to be encoded
            response: (str or iterable of str) This contains the response that is required to be encoded
            margin: (float) margin tuning parameter for triplet / contrastive loss
            loss: (str) name of loss function. ('cosine', 'contrastive', 'triplet'). Default setting is 'triplet.
            context: (str or iterable of str) Ignored for BERT/ALBERT
            neg_answer: (str or iterable of str) This contains the distractor responses that is required to be encoded
            neg_answer_context: (str or iterable of str) Ignored for BERT/ALBERT
            label: (list of int) This contain the label for contrastive loss

        return:
            loss_value: (float) This returns the loss of the training task

        """
        question_id_mask_seg = preprocess_str(
            question, self.max_seq_length,
            self.tokenizer
        )

        response_id_mask_seg = preprocess_str(
            answer, self.max_seq_length,
            self.tokenizer
        )

        # for eager execution finetuning
        with tf.GradientTape() as tape:

            # tf-hub's keras layer can take the lists directly
            # but the bert_model object needs the inputs to be tf.constants
            question_embeddings, q_sequence_output = self.bert_model.get_layer(self.bert_layer_name)(question_id_mask_seg)
            response_embeddings = self.bert_model(
                [tf.constant(response_id_mask_seg[0]),
                 tf.constant(response_id_mask_seg[1]),
                 tf.constant(response_id_mask_seg[2])]
            )

            if loss == 'cosine':
                self.cost = tf.keras.losses.CosineSimilarity(axis=1)
                cost_value = self.cost(question_embeddings,
                                       response_embeddings)

            elif loss == 'contrastive':
                """
                https://www.tensorflow.org/addons/api_docs/python/tfa/losses/ContrastiveLoss

                y_true to be a vector of binary labels
                y_hat to be the respective distances
                """
                self.cosine_dist = tf.keras.losses.CosineSimilarity(axis=1)
                cosine_dist_value = self.cosine_dist(question_embeddings,
                                                     response_embeddings)

                self.cost = tfa.losses.contrastive.ContrastiveLoss(margin=margin)
                cost_value = self.cost(label, cosine_dist_value)

            elif loss == 'triplet':
                """
                Triplet loss uses a non-official self-implementated loss function outside of TF based on cosine distance
                """
                # encode the negative response
                neg_answer_id_mask_seg = preprocess_str(
                    neg_answer,
                    self.max_seq_length,
                    self.tokenizer
                )

                neg_response_embeddings = self.bert_model(
                    [tf.constant(neg_answer_id_mask_seg[0]),
                     tf.constant(neg_answer_id_mask_seg[1]),
                     tf.constant(neg_answer_id_mask_seg[2])])

                cost_value = triplet_loss(question_embeddings,
                                          response_embeddings,
                                          neg_response_embeddings)

            # record loss
            self.cost_history.append(cost_value.numpy().mean())

        # apply gradient
        self.grads = tape.gradient(cost_value, self.var_finetune)
        self.optimizer.apply_gradients(zip(self.grads, self.var_finetune))

        return cost_value.numpy().mean()

    def save_weights(self, save_dir=None):
        '''
        Save the BERT model into a directory

        The original saving procedure taken from:
        https://github.com/tensorflow/models/blob/master/official/nlp/bert/export_tfhub.py
        The tf-hub module does includes the str vocab_file directory and do_lower_case boolean
        The future restore() function should depend on a fresh copy of the vocab file,
        because loading the model in a different directory demands a different directory for the vocab.
        However, the string vocab_file directory and do_lower_case boolean is kept to the saved model anyway
        '''

        self.bert_model.vocab_file = self.vocab_file
        self.bert_model.do_lower_case = self.do_lower_case

        self.bert_model.save(save_dir, include_optimizer=False)

    def restore_weights(self, save_dir=None):
        """
        Load saved model from savepath

        hub.KerasLayer is unrecognized by tf.keras' save and load_model methods.
        The trick is to feed a custom_objects dict object
        This solution given by qo-o-op in this link:
        https://github.com/tensorflow/tensorflow/issues/26835

        Args:
            savepath: (str) dir path of the 
        """
        self.bert_model = tf.keras.models.load_model(
            save_dir,
            custom_objects={'KerasLayer': hub.KerasLayer}
        )

        self.init_signatures()
