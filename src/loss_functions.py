import tensorflow as tf


def triplet_loss(anchor_vector, positive_vector, negative_vector, metric='cosine_dist', margin=0.009):
    """Computes the triplet loss with semi-hard negative mining.
    The loss encourages the positive distances (between a pair of embeddings with
    the same labels) to be smaller than the minimum negative distance among
    which are at least greater than the positive distance plus the margin constant
    (called semi-hard negative) in the mini-batch. If no such negative exists,
    uses the largest negative distance instead.
    See: https://arxiv.org/abs/1503.03832.

    :type anchor_vector: tf.Tensor
    :type positive_vector: tf.Tensor
    :type negative_vector: tf.Tensor
    :type metric: str
    :type margin: float
    :param anchor_vector: The anchor vector in this use case should be the encoded query. 
    :param positive_vector: The positive vector in this use case should be the encoded response. 
    :param negative_vector: The negative vector in this use case should be the wrong encoded response. 
    :param metric: Specify loss function
    :param margin: Margin parameter in loss function. See link above. 
    :return: the triplet loss value, as a tf.float32 scalar.
    """
    cosine_distance = tf.keras.losses.CosineSimilarity(axis=1)
    d_pos = cosine_distance(anchor_vector, positive_vector)
    d_neg = cosine_distance(anchor_vector, negative_vector)
    loss = tf.maximum(0., margin + d_pos - d_neg)
    loss = tf.reduce_mean(loss)
    return loss

