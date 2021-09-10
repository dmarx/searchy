# semantic image search CLI thingy

# CLI usage (build tests around these use cases)

# searchy init . # create a new index here using the default name
# searchy init -n database.srchy
# searchy process path/to/images/root/ # add photos to index
# searchy get 'text prompt'
# searchy get 'text prompt' -n 10

import CLIP
import faiss
import sqlite
import torch # dataloader

# fire cli?
# https://github.com/google/python-fire
class SearchyClient:
    def __init__(self, cfg=None):
        if config:
            self.from_config()
        self._get_vector_store() #create_FAISS()
        self._get_filepath_db() #create_sqlite() # might not be necessary
        
    def _get_vector_store(self):
        pass
    
    def _get_filepath_db(self):
        pass

    def from_config(self):
        pass
        # maybe write database metadata to a file
        # e.g. configs for transforms, file paths, etc.
        # specify where to load model from, where to download to, how to load (e.g. from torch hub)

    def process(self, path='.'):
        for batch in get_batches():
            augment_images()
        pass
    
    def get(self, query, path='.', n=3, return_scores=True):
        queries = augment(query)
        text_vectors = encode_text(queries)
        knn(text_vectors)
        # tqdm, colors, spinnies, pprint
        pass
        
# ML Idea: approximate nearest neighbors via randomly initialized trees
# - (with respect to choosing splitting features) 
# - tree encoding dimensionality reduction
