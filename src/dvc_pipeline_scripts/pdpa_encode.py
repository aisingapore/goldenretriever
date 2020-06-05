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
    responses = df['ans_str'].drop_duplicates().tolist()
    contexts = [' '] * len(responses)
    return responses, contexts 


def extract_response_embeddings(responses, contexts, encoder=None, save_dir=None):
    if encoder == 'BERTEncoder':
        from src.encoders import BERTEncoder
        enc = BERTEncoder()
    elif encoder == 'ALBERTEncoder':
        from src.encoders import ALBERTEncoder
        enc = ALBERTEncoder()
    else:
        from src.encoders import USEEncoder
        enc = USEEncoder()

    gr = GoldenRetriever(enc)
    if save_dir is not None:
        gr.restore_encoder(save_dir=save_dir)
    embs = gr.encoder.encode(responses, contexts, string_type='response')
    print(f"responses created with dim {len(embs[0])}")
    return embs


def index_responses(responses, embs, output_folder, index_prefix):
    """Index responses w SimpleNNIndex class"""
    simple_nn = SimpleNNIndex(len(embs[0]))
    simple_nn.build(responses, embs)
    simple_nn.save(output_folder + '/' + index_prefix)
    print(f'index saved to {output_folder}')


@click.command()
@click.option('--data', help='path to file with raw responses')
@click.option('--output_folder', help='path to save index')
@click.option('--index_prefix', help='prefix to add to -data.pkl of index')
@click.option('--gr_model', help='name of gr model')
def main(data, output_folder, index_prefix, gr_model):
    responses, contexts = get_responses_and_contexts(data)
    embs = extract_response_embeddings(responses, contexts, gr_model)
    index_responses(responses, embs, output_folder, index_prefix)


if __name__=='__main__':
    main()
