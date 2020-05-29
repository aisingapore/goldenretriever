import pytest
import pandas as pd
import os
import random
import numpy as np
import shutil

from mock import Mock
from sklearn.model_selection import train_test_split

from src.models import GoldenRetriever
from src.encoders import USEEncoder
from src.data_handler.kb_handler import kb, kb_handler, generate_mappings


def load_es_kb():
    kbs = []

    responses = [
        "Personal data refers to data, whether true or not...",
        "The PDPA was implemented in phases to allow time for organisations to...",
        "The PDPA aims to safeguard individuals personal data...",
        "The PDPA will strengthen Singapore's overall economic competitiveness...",
        "The provisions of the PDPA were formulated keeping in mind"
    ]

    queries = [
        "What is personal data?",
        "When did the PDPA come into force?",
        "What are the objectives of the PDPA?",
        "How does the PDPA benefit business?",
        "How will the PDPA impact business costs?"
    ]

    query_id = [0, 1, 2, 3, 4]
    clause_id = [0, 1, 2, 3, 4]
    d = {
        "clause_id": clause_id, "raw_string": responses, "processed_string": responses,
        "context_string": responses, "query_string": queries, "query_id": query_id
    }

    df = pd.DataFrame(d)

    mappings = generate_mappings(df.processed_string, df.query_string)

    responses_df = df.loc[:, ['clause_id', 'raw_string', 'context_string']]
    queries_df = df.loc[:, ['query_id', 'query_string']]
    nrf = kb('nrf', responses_df, queries_df, mappings)
    kbs.append(nrf)
    return kbs


def test_make_query(monkeypatch):

    def mock_load_es_kb(*args, **kwargs):
        return load_es_kb()

    monkeypatch.setattr(kb_handler, 'load_es_kb', mock_load_es_kb)
    use = USEEncoder()
    gr = GoldenRetriever(use)

    kbh = kb_handler()
    kbs = kbh.load_es_kb()

    gr.load_kb(kbs)

    querystring = "Can I change funding source"
    actual = gr.make_query(querystring, top_k=5, index=False, predict_type="query", kb_name="nrf")

    assert isinstance(actual[0], list)
    assert isinstance(actual[0][0], str)
    assert len(actual[0]) == 5


def gen(batch_size, query, response, neg_response, shuffle_data=False):
    random.seed(42)
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


def _generate_neg_ans(df, train_dict):
    """
    Generates negative answer from dataframe by randomization
    
    Returns a dict, with keys pointing to each kb, pointing to 
    2 arrays of indices, one of correct answers and one of wrong answers,
    generated randomly

    Sample output:
    --------------
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
    random.seed(42)

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


def random_triplet_generator(df, train_dict):
    train_dict_with_neg = _generate_neg_ans(df, train_dict)
    train_pos_idxs = np.concatenate([v[0] for k,v in train_dict_with_neg.items()], axis=0)
    train_neg_idxs = np.concatenate([v[1] for k,v in train_dict_with_neg.items()], axis=0)

    train_query = df.iloc[train_pos_idxs].query_string.tolist()
    train_response = df.iloc[train_pos_idxs].processed_string.tolist()
    train_neg_response = df.iloc[train_neg_idxs].processed_string.tolist()

    train_dataset_loader = gen(32, train_query, train_response, train_neg_response, shuffle_data=True)

    return train_dataset_loader


@pytest.fixture
def create_delete_model_savepath():

    savepath = os.path.join(os.getcwd(), "finetune")
    os.makedirs(savepath)
    yield savepath

    shutil.rmtree(savepath)


def test_finetune_export_restore(monkeypatch, create_delete_model_savepath):

    def mock_load_es_kb(*args, **kwargs):
        return load_es_kb()

    monkeypatch.setattr(kb_handler, 'load_es_kb', mock_load_es_kb)
    use = USEEncoder()
    gr = GoldenRetriever(use)

    kbh = kb_handler()
    kbs = kbh.load_es_kb()

    train_dict = dict()
    test_dict = dict()

    df = pd.concat([single_kb.create_df() for single_kb in kbs]).reset_index(drop='True')
    kb_names = df['kb_name'].unique()

    for kb_name in kb_names:
        kb_id = df[df['kb_name'] == kb_name].index.values
        train_idx, test_idx = train_test_split(kb_id, test_size=0.4,
                                            random_state=100)

        train_dict[kb_name] = train_idx
        test_dict[kb_name] = test_idx

    train_dataset_loader = random_triplet_generator(df, train_dict)

    for i in range(1):
        cost_mean_total = 0
        batch_counter = 0

        for q, r, neg_r in train_dataset_loader:
            
            # Test using 1 batch of training data            
            if batch_counter == 1:
                break

            cost_mean_batch = gr.finetune(
                question=q, answer=r, context=r,
                neg_answer=neg_r, neg_answer_context=neg_r,
                margin=0.3, loss="triplet"
            )

            cost_mean_total += cost_mean_batch

            batch_counter += 1

    initial_pred = gr.predict("What is personal data?")
    save_dir = create_delete_model_savepath
    gr.export_encoder(save_dir=save_dir)

    use_new = USEEncoder()
    gr_new = GoldenRetriever(use_new)
    gr_new.restore_encoder(save_dir=save_dir)
    restored_pred = gr_new.predict("What is personal data?")

    assert isinstance(cost_mean_total, np.floating)
    assert cost_mean_total != 0.0000
    assert os.path.isdir(save_dir)
    assert np.array_equal(initial_pred, restored_pred)
