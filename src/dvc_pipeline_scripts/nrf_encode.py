import pandas as pd 
import click 
from src.models import GoldenRetriever
from src.encoders import USEEncoder, ALBERTEncoder, BERTEncoder
from src.prebuilt_index import SimpleNNIndex

"""
example script of how to vectorize responses in KB and save them into a simple neighbors index 
to be used at query time 
"""

def get_responses_and_contexts(filepath):
    """Read csv file and return responses and contexts"""
    df = pd.read_csv(filepath)
    responses = df['ans_str'].tolist()
    contexts = [' '] * len(responses)
    return responses, contexts 


def extract_response_embeddings(gr_model, responses, contexts):
    """Extract encoded responses"""
    if gr_model == 'BERTEncoder':
        enc = BERTEncoder()
    elif gr_model == 'ALBERTEncoder':
        enc = ALBERTEncoder()
    else:
        enc = USEEncoder()
    gr = GoldenRetriever(enc)
    embs = gr.encoder.encode(responses, contexts, string_type="response")
    print(f"responses created with dim {len(embs['outputs'][0])}")
    return embs


def index_responses(responses, embs, output_folder, index_prefix):
    """Index responses w SimpleNNIndex class"""
    simple_nn = SimpleNNIndex(len(embs['outputs'][0]))
    simple_nn.build(responses, embs)
    simple_nn.save(output_folder, index_prefix)


@click.command()
@click.option('--data', help='path to file with raw responses')
@click.option('--output_folder', help='path to save index')
@click.option('--index_prefix', help='prefix to add to -data.pkl of index')
@click.option('--gr_model', help='name of gr model')
def main(data, output_folder, index_prefix, tfhub_module_url):
    responses, contexts = get_responses_and_contexts(data)
    embs = extract_response_embeddings(tfhub_module_url, responses, contexts)
    index_responses(responses, embs, output_folder, index_prefix)


if __name__=='__main__':
    main()
