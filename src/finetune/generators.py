"""
TRIPLET GENERATORS:
Functions to generate training triplets for triplet loss to finetune GR
"""
import random

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def _generate_neg_ans(df, train_dict, CONFIG):
    """
    Generates negative answer from dataframe by randomization

    :type df: pd.DataFrame
    :type train_dict: dict
    :type CONFIG: config object
    :param df: Contains the query response pair
    :param train_dict: Contains the indices of train test pairs
    :param CONFIG: config object
    :return: a dict, with keys pointing to each kb, pointing to 2 arrays of indices, one of correct answers and one of wrong answers, generated randomly
    
    Sample output:
    .. Highlight:: python
    .. Code-block:: python
        {'PDPA': [array([ 95,  84,  42, 185, 187, 172, 145,  71,   5,  36,  43, 153,  70,
                        66,  53,  98, 180,  94, 138, 176,  79,  87, 103,  67,  24,   8]),
                array([141, 129, 155,   5, 108, 180,  63,   0, 143, 130,  98, 132,  61,
                        138,  24, 187,  86, 153,  94, 140, 162, 109,  56, 105, 185, 165])],
        'nrf': [array([214, 240, 234, 235, 326, 244, 226, 252, 317, 331, 259, 215, 333,
                        283, 299, 263, 220, 204]),
                array([249, 245, 331, 290, 254, 249, 249, 261, 296, 251, 214, 240, 275,
                        210, 223, 259, 212, 205])]}
"""
    train_dict_with_neg = {}
    random.seed(CONFIG.random_seed)

    for kb, ans_pos_idxs in train_dict.items():
        keys = []
        shuffled_ans_pos_idxs = ans_pos_idxs.copy()
        random.shuffle(shuffled_ans_pos_idxs)
        ans_neg_idxs = shuffled_ans_pos_idxs.copy()

        correct_same_as_wrong = df.loc[ans_neg_idxs, 'processed_string'].values == df.loc[ans_pos_idxs, 'processed_string'].values
        while sum(correct_same_as_wrong) > 0:
            random.shuffle(shuffled_ans_pos_idxs)
            ans_neg_idxs[correct_same_as_wrong] = shuffled_ans_pos_idxs[correct_same_as_wrong]
            correct_same_as_wrong = df.loc[ans_neg_idxs, 'processed_string'].values == df.loc[ans_pos_idxs, 'processed_string'].values

        keys.append(ans_pos_idxs)
        keys.append(np.array(ans_neg_idxs))

        train_dict_with_neg[kb] = keys 
    
    return train_dict_with_neg

def _generate_hard_neg_ans(df, train_dict, model, CONFIG):
    """
    Generates negative answer from dataframe by based on model selection.

    :type df: pd.DataFrame
    :type train_dict: dict
    :type CONFIG: config object
    :param df: Contains the query response pairs
    :param train_dict: Contains the indices of train test pairs
    :param CONFIG: config object
    :param model: GoldenRetriever's Model class object
    :return: a dict, with keys pointing to each kb, pointing to 2 arrays of indices, one of correct answers and one of wrong answers, generated randomly
    
    Sample output:
    .. Highlight:: python
    .. Code-block:: python
        {'PDPA': [array([ 95,  84,  42, 185, 187, 172, 145,  71,   5,  36,  43, 153,  70,
                        140, 165,   0,  78, 162,  68, 184, 179,  30, 106,  13,  72,  17,
                        18,  38, 109,  47, 113,  56,  27,  63, 147, 105, 121,   2,  80,
                        182,  61,  49, 135, 193,  91,   4, 100, 141, 129, 159, 132, 108,
                        155, 130,  86,  93, 137, 144,  58,  60, 107, 143, 194,  34,  14,
                        66,  53,  98, 180,  94, 138, 176,  79,  87, 103,  67,  24,   8]),
                array([141, 129, 155,   5, 108, 180,  63,   0, 143, 130,  98, 132,  61,
                        103, 137,  13,  17,  71, 107, 144, 121,  68,  66, 184, 179, 135,
                        113, 194,  58,  53, 193,  34,  42,  78,  60, 106, 182,  72, 172,
                        145, 100, 176,  36, 159,  30,  14,  93,  43,  95,  79,   2,  87,
                        8,  18, 147,  91,  49,   4,  70,  67,  84,  80,  27,  47,  38,
                        138,  24, 187,  86, 153,  94, 140, 162, 109,  56, 105, 185, 165])],
        'nrf': [array([214, 240, 234, 235, 326, 244, 226, 252, 317, 331, 259, 215, 333,
                        318, 276, 267, 251, 329, 257, 261, 243, 245, 203, 337, 255, 287,
                        315, 296, 279, 209, 197, 227, 200, 304, 223, 198, 282, 289, 205,
                        319, 212, 254, 256, 303, 338, 230, 210, 262, 249, 294, 290, 275,
                        283, 299, 263, 220, 204]),
                array([249, 245, 331, 290, 254, 249, 249, 261, 296, 251, 214, 240, 275,
                        294, 319, 337, 215, 197, 200, 257, 289, 203, 282, 252, 315, 317,
                        230, 283, 304, 279, 333, 249, 299, 204, 318, 326, 262, 287, 256,
                        234, 303, 235, 243, 276, 198, 338, 220, 329, 255, 209, 263, 267,
                        210, 223, 259, 212, 205])]}
    """
    train_dict_with_neg = {}
    random.seed(CONFIG.random_seed)

    for kb, ans_pos_idxs in train_dict.items():
        keys = []
        train_df = df.loc[ans_pos_idxs]

        # encodings of all possible answers
        all_possible_answers_in_kb = train_df.processed_string.unique().tolist()
        encoded_all_possible_answers_in_kb = model.predict(all_possible_answers_in_kb, string_type='response')

        # encodings of train questions
        train_questions = train_df.query_string
        encoded_train_questions = model.predict(train_questions, string_type='query')

        # get similarity matrix
        similarity_matrix = cosine_similarity(encoded_train_questions, encoded_all_possible_answers_in_kb)

        # get index of correct answers, indexed according to unique answers
        correct_answers = train_df.processed_string.tolist()
        idx_of_correct_answers = [all_possible_answers_in_kb.index(correct_answer) for correct_answer in correct_answers]

        # get second best answer index by kb_df
        ans_neg_idxs = []
        for idx_of_correct_answer, similarity_array in zip(idx_of_correct_answers, similarity_matrix):
            similarity_array[idx_of_correct_answer] = -1
            second_best_answer_idx_in_all_possible_answers = similarity_array.argmax()
            second_best_answer_string = all_possible_answers_in_kb[second_best_answer_idx_in_all_possible_answers]
            second_best_answer_idx_in_kb_df = train_df.loc[train_df.processed_string == second_best_answer_string].index[0]
            ans_neg_idxs.append(second_best_answer_idx_in_kb_df)

        # return a list of correct and close wrong answers
        keys.append(ans_pos_idxs)
        keys.append(np.array(ans_neg_idxs))
        train_dict_with_neg[kb] = keys 
    
    return train_dict_with_neg

def gen(query, response, neg_response, CONFIG, shuffle_data=False):
    """
    Create a generator that of queries, responses and negative responses.
    """
    batch_size = CONFIG.train_batch_size
    random.seed(CONFIG.random_seed)
    zip_list = list(zip(query,response,neg_response))

    num_samples = len(query)
    while True:
        if shuffle_data:
            random.shuffle(zip_list)

        for offset in range(0, num_samples, batch_size):
            q_batch = [x[0] for x in zip_list[offset:offset+batch_size]]
            r_batch = [x[1] for x in zip_list[offset:offset+batch_size]]
            neg_r_batch = [x[2] for x in zip_list[offset:offset+batch_size]]
        
            yield(q_batch, r_batch, neg_r_batch)

def random_triplet_generator(df, train_dict, CONFIG):
    """
    Returns a generator that gives batches of training triplets
    """
    train_dict_with_neg = _generate_neg_ans(df, train_dict, CONFIG)
    train_pos_idxs = np.concatenate([v[0] for k,v in train_dict_with_neg.items()], axis=0)
    train_neg_idxs = np.concatenate([v[1] for k,v in train_dict_with_neg.items()], axis=0)

    train_query = df.iloc[train_pos_idxs].query_string.tolist()
    train_response = df.iloc[train_pos_idxs].processed_string.tolist()
    train_neg_response = df.iloc[train_neg_idxs].processed_string.tolist()
    
    print("train batch size: {CONFIG.train_batch_size}")
    train_dataset_loader = gen(train_query, train_response, train_neg_response, CONFIG, shuffle_data=True)
    
    return train_dataset_loader

def hard_triplet_generator(df, train_dict, model, CONFIG):
    """
    Returns a generator that gives batches of training triplets
    """
    train_dict_with_neg = _generate_hard_neg_ans(df, train_dict, model, CONFIG)
    train_pos_idxs = np.concatenate([v[0] for k,v in train_dict_with_neg.items()], axis=0)
    train_neg_idxs = np.concatenate([v[1] for k,v in train_dict_with_neg.items()], axis=0)

    train_query = df.iloc[train_pos_idxs].query_string.tolist()
    train_response = df.iloc[train_pos_idxs].processed_string.tolist()
    train_neg_response = df.iloc[train_neg_idxs].processed_string.tolist()
    
    train_dataset_loader = gen(train_query, train_response, train_neg_response, CONFIG, shuffle_data=True)
    
    return train_dataset_loader