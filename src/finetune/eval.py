"""
Functions to evaluate model
These includes metrics and other utility functions
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats.mstats import rankdata


def mrr(ranks):
    """
    Calculate mean reciprocal rank
    Function taken from: https://github.com/google/retrieval-qa-eval/blob/master/squad_eval.py

    :type ranks: list
    :param ranks: predicted ranks of the correct responses 
    :return: float value containing the MRR 
    """
    return sum([1/v for v in ranks])/len(ranks)

def recall_at_n(ranks, n=3):
    """
    Calculate recall @ N
    Function taken from: https://github.com/google/retrieval-qa-eval/blob/master/squad_eval.py

    :type ranks: list
    :param ranks: predicted ranks of the correct responses 
    :return: float value containing the Recall@N
    """
    num = len([rank for rank in ranks if rank <= n])
    return num / len(ranks)

def get_eval_dict(ranks):
    """
    Score the predicted ranks according to various metricss

    :type ranks: list
    :param ranks: predicted ranks of the correct responses 
    :return: dict that contains the metrics and their respective keys
    """
    eval_dict = {}
    eval_dict['mrr_score'] = mrr(ranks)
    eval_dict['r1_score'] = recall_at_n(ranks, 1)
    eval_dict['r2_score'] = recall_at_n(ranks, 2)
    eval_dict['r3_score'] = recall_at_n(ranks, 3)
    return eval_dict


def eval_model(model, df, test_dict):
    """
    Evalutate golden retriever object

    :type model: GoldenRetriever's Model class object
    :type df: pd.DataFrame
    :type test_dict: dict
    :param model:GoldenRetriever's Model class object
    :param df: Contains the query response pairs
    :param test_dict: pContains the indices of train test pairs
    :return overall_eval: pd.DataFrame that contains the metrics
    :return eval_dict: dict of the same metrics
<<<<<<< HEAD

    Sample output:
    .. Highlight:: python
    .. Code-block:: python
                                    mrr_score  r1_score  r2_score  r3_score
        PDPA                        0.640719  0.525424  0.627119  0.720339
        nrf                         0.460211  0.275862  0.482759  0.528736
        critical-illness-insurance  0.329302  0.178571  0.342857       0.4
        other-insurance             0.474588  0.259259  0.444444  0.611111
        Steam_engine                0.689601  0.550388  0.744186  0.775194
        1973_oil_crisis             0.781951   0.65625   0.84375  0.890625
        Across_all_kb               0.551312  0.402027  0.570946  0.636824
=======
>>>>>>> 043a6d5f6cb25f97b8800a6217a30f7065ebe853
    """
    eval_dict = {}

    for kb_name in df.kb_name.unique():

        # dict stores eval metrics and relevance ranks
        eval_kb_dict = {}  
        # test-mask is a int array
        # that chooses specific test questions
        # e.g.  test_mask [True, True, False]
        #       query_idx = [0,1]
        kb_df = df.loc[df.kb_name == kb_name]
        kb_idx = df.loc[df.kb_name == kb_name].index
        test_mask = np.isin(kb_idx, test_dict[kb_name])
        # test_idx_mask = np.arange(len(kb_df))[test_mask]

        # get string queries and responses, unduplicated as a list
        kb_df = kb_df.reset_index(drop=True)
        query_list = kb_df.query_string.tolist()
        response_list_w_duplicates = kb_df.processed_string.tolist()
        response_list = kb_df.processed_string.drop_duplicates().tolist() 
        # this index list is important
        # it lists the index of the correct answer for every question
        # e.g. for 20 questions mapped to 5 repeated answers
        # it has 20 elements, each between 0 and 4
        response_idx_list = [response_list.index(nonunique_response_string) 
                            for nonunique_response_string in response_list_w_duplicates]
        response_idx_list = np.array(response_idx_list)[[test_mask]]
        
        encoded_queries = model.predict(query_list, string_type='query')
        encoded_responses = model.predict(response_list, string_type='response')

        # get matrix of shape [Q_test x Responses]
        # this holds the relevance rankings of the responses to each test ques
        test_similarities = cosine_similarity(encoded_queries[test_mask], encoded_responses)
        answer_ranks = test_similarities.shape[-1] - rankdata(test_similarities, axis=1) + 1

        # ranks_to_eval
        ranks_to_eval = [answer_rank[correct_answer_idx] 
                        for answer_rank, correct_answer_idx 
                        in zip( answer_ranks, response_idx_list )]


        # get eval metrics -> eval_kb_dict 
        # store in one large dict -> eval_dict
        eval_kb_dict = get_eval_dict(ranks_to_eval)
        eval_kb_dict['answer_ranks'] = answer_ranks
        eval_kb_dict['ranks_to_eval'] = ranks_to_eval
        eval_dict[kb_name] = eval_kb_dict.copy()


    # overall_eval is a dataframe that 
    # tracks performance across the different knowledge bases
    # but individually
    overall_eval = pd.DataFrame(eval_dict).T.drop(['answer_ranks', 'ranks_to_eval'], axis=1)

    # Finally we get eval metrics for across all different KBs
    correct_answer_ranks_across_kb = []
    for key in eval_dict.keys():
        correct_answer_ranks_across_kb.extend(eval_dict[key]['ranks_to_eval'])
        
    # get eval metrics across all knowledge bases combined
    across_kb_scores = get_eval_dict(correct_answer_ranks_across_kb)
    across_kb_scores_ = {'Across_all_kb':across_kb_scores}
    across_kb_scores_ = pd.DataFrame(across_kb_scores_).T

    overall_eval = pd.concat([overall_eval,across_kb_scores_])
    return overall_eval, eval_dict