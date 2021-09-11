# semantic image search CLI thingy

# CLI usage (build tests around these use cases)

# searchy init . # create a new index here using the default name
# searchy init -n database.srchy
# searchy process path/to/images/root/ # add photos to index
# searchy get 'text prompt'
# searchy get 'text prompt' -n 10

#import CLIP
#import faiss
import sqlite3
import torch # dataloader

#https://huggingface.co/transformers/model_doc/clip.html
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIP(torch.nn.Module):
    def __init__(self, model_string="openai/clip-vit-base-patch32"):
        super().__init__()
        self._model_string = model_string
        self.model = CLIPModel.from_pretrained(model_string)
        self.processor = CLIPProcessor.from_pretrained(model_string)
    

# fire cli?
# https://github.com/google/python-fire
class SearchyClient:
    def __init__(self, cfg=None):
        if cfg:
            self.from_config()
        self._get_vector_store() #create_FAISS()
        self._get_filepath_db() #create_sqlite() # might not be necessary
        self.clip = CLIP()
        
    def _get_vector_store(self):
        pass
    
    def _get_filepath_db(self):
        pass

    def from_config(self):
        pass
        # maybe write database metadata to a file
        # e.g. configs for transforms, file paths, etc.
        # specify where to load model from, where to download to, how to load (e.g. from torch hub)
    
    def load_image(self, path):
        return Image.open(path)

    def process(self, path='.'):
        im = self.load_image(path)
        return self.clip.processor(images=im)
        #for batch in get_batches():
        #    augment_images()
        #pass
    
    def get(self, query, path='.', n=3, return_scores=True):
        queries = augment(query)
        text_vectors = encode_text(queries)
        knn(text_vectors)
        # tqdm, colors, spinnies, pprint
        pass
        
# ML Idea: approximate nearest neighbors via randomly initialized trees
# - (with respect to choosing splitting features) 
# - tree encoding dimensionality reduction
