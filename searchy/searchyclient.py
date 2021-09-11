# semantic image search CLI thingy

# CLI usage (build tests around these use cases)

# searchy init . # create a new index here using the default name
# searchy init -n database.srchy
# searchy process path/to/images/root/ # add photos to index
# searchy get 'text prompt'
# searchy get 'text prompt' -n 10

#import CLIP
#import faiss
from pathlib import Path
import imghdr # builtin
import sqlite3

#https://huggingface.co/transformers/model_doc/clip.html
from PIL import Image
import torch # dataloader
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel


class CLIP(torch.nn.Module):
    def __init__(self, model_string="openai/clip-vit-base-patch32"):
        super().__init__()
        self._model_string = model_string
        self.model = CLIPModel.from_pretrained(model_string)
        self.processor = CLIPProcessor.from_pretrained(model_string)
    def project_images(self, images, normalize=True):
        imgs = self.processor(images=images, return_tensors="pt", padding=True)
        feats = self.model.get_image_features(**imgs)
        if normalize:
            feats = self.normalize(feats)
        return feats
    def project_texts(self, texts, normalize=True):
        txts = self.processor(text=texts, return_tensors="pt", padding=True)
        feats = self.model.get_text_features(**txts)
        if normalize:
            feats = self.normalize(feats)
        return feats
    def normalize(self, x):
        return x / x.norm(dim=-1, keepdim=True)
    


# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
# NB: might be able to use lightning datamodule to parallelize data processing
class RecusiveImagesPath(Dataset):
    def __init__(self, root='sample_data/images'):
        self.root = root
        self.get_image_paths()
    def get_image_paths(self):
        self.img_paths = []
        for path_obj in Path(self.root).glob('*'):
            if imghdr.what(path_obj) is not None:
                self.img_paths.append(path_obj)
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        path = self.img_paths[idx]
        return Image.open(path), path

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
