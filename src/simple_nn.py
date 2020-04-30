import pandas as pd 
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tensorflow_text
import simpleneighbors

nrf = pd.read_csv('data/nrf.csv')
nrf.head()

queries = nrf['query_str'].fillna('na').tolist()
responses = nrf['ans_str'].tolist()
contexts = [''] * len(responses)

print('loading model')
module = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3')

print('computing embeddings')

question_embeddings = module.signatures['question_encoder'](
            tf.constant(queries))
response_embeddings = module.signatures['response_encoder'](
        input=tf.constant(responses),
        context=tf.constant(contexts))
print('embeddings done')

np.inner(question_embeddings['outputs'], response_embeddings['outputs'])

print('building index')
index = simpleneighbors.SimpleNeighbors(len(response_embeddings['outputs'][0]), metric='angular')
resp_emb_tup = list(zip(responses, response_embeddings['outputs'].numpy()))
index.feed(resp_emb_tup)
index.build()
print(index.nearest(question_embeddings['outputs'][0], 1))
index.save('nrf')
print('index saved')