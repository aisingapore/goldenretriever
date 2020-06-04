from simpleneighbors import SimpleNeighbors
import pickle


class SimpleNNIndex(SimpleNeighbors):
    """
    Simple Neighbors Index for calculating similarity between queries and reponses vectorized by Golden Retriever

    This class wraps the SimpleNeighbors python package. SimpleNeighbors will select a backend implementation 
    depending on what packages are available in your environment. Therefore it is recommended that you install Annoy 
    ``pip install annoy`` to enable the Annoy backend. 

    :params emb_dim_size: number of dimensions in the data (eg. 512)
    :param metric: distance metric to use. Default is 'angular', which is an approximation of cosine distance
    """
    def __init__(self, emb_dim_size, metric='angular'):
        super().__init__(emb_dim_size, metric)

    def build(self, sentences, sentence_embeddings):
        """
        builds precomputed vector index from QA responses. uses the Annoy library by default. 
        
        :param sentences: responses in string form
        :param sentence_embeddings: responses in embedding form
        :return: simpleneighbors index for nearest neighbors vector lookup
        """
        sentence_emb_tup = list(zip(sentences, sentence_embeddings['outputs'].numpy()))
        super().feed(sentence_emb_tup)
        super().build()
        print('index built')
    
    def query(self, query_embeddings, num_nbrs):
        """
        finds response closest to the query vector

        The query vector should have the same number of dimensions as the dimensions of the index.
        Search is limited to the given number of items. Results are given in order of proximity. 
        :param query_embeddings: query in embedding form
        :param num_nbrs: number of results to return
        :return: list of items sorted by pro
        """
        return super().nearest(query_embeddings, num_nbrs)

    def save(self, index_prefix):
        """
        saves index to disk. With the Annoy backend, there are two files produced: 
        the serialized Annoy index and a pickle with other data from the object

        :param prefix: filename prefix for the Annoy index and object data 
        :return: None
        """
        super().save(index_prefix)
        print('index saved')

    @classmethod
    def load(cls, prefix):
        """
        restores a previously-saved index

        :param prefix: prefix used when saving index
        :return: SimpleNNIndex object restored from specified files
        """

        with open(prefix + '-data.pkl', 'rb') as fh:
            data = pickle.load(fh)
        new_index = cls(emb_dim_size=data['dims'], metric=data['metric'])
        new_index.id_map = data['id_map']
        new_index.corpus = data['corpus']
        new_index.i = data['i']
        new_index.built = data['built']
        new_index.backend.load(prefix + ".idx")        
        return new_index