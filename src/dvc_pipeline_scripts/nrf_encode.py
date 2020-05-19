import pandas as pd 
import click 
from src.encoders import USEResponseEncoder 
from src.prebuilt_index import SimpleNNIndex

"""
example script of how to vectorize responses in KB and save them into a simple neighbors index 
to be used at query time 
"""


def get_responses_and_contexts(filepath):
    df = pd.read_csv(filepath)
    responses = df['ans_str'].tolist()[0:5]
    contexts = [' '] * len(responses)
    return responses, contexts 


def extract_response_embeddings(tfhub_module_url, responses, contexts):
    embs = USEResponseEncoder(tfhub_module_url).encode(responses, contexts)
    print(f"responses created with dim {len(embs['outputs'][0])}")
    return embs


def index_responses(responses, embs, output_folder, index_prefix):
    simple_nn = SimpleNNIndex(len(embs['outputs'][0]))
    simple_nn.build(responses, embs)
    simple_nn.save(output_folder, index_prefix)


@click.command()
@click.option('--data', help='path to file with raw responses')
@click.option('--output_folder', help='path to save index')
@click.option('--index_prefix', help='prefix to add to -data.pkl of index')
@click.option('--tfhub_module_url', help='link to tensorflowhub model')
def main(data, output_folder, index_prefix, tfhub_module_url):
    responses, contexts = get_responses_and_contexts(data)
    embs = extract_response_embeddings(tfhub_module_url, responses, contexts)
    index_responses(responses, embs, output_folder, index_prefix)


if __name__=='__main__':
    main()
