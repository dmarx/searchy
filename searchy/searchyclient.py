# semantic image search CLI thingy

# CLI usage (build tests around these use cases)

# searchy init . # create a new index here using the default name
# searchy init -n database.srchy
# searchy process path/to/images/root/ # add photos to index
# searchy get 'text prompt'
# searchy get 'text prompt' -n 10

import imghdr # builtin
from pathlib import Path
import sqlite3

import faiss
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
#from torchvision.datasets import ImageFolder
from torchvision.io import read_image
from transformers import CLIPProcessor, CLIPModel
from torchvision.transforms import ToTensor
from torchvision import transforms


class CLIP(torch.nn.Module):
    n_feats = 5125
    def __init__(self, model_string="openai/clip-vit-base-patch32"):
        super().__init__()
        self._model_string = model_string
        self.model = CLIPModel.from_pretrained(model_string)
        self.processor = CLIPProcessor.from_pretrained(model_string)
        self.model.eval()
    def project_images(self, images, normalize=True, preprocessed=False):
        if not preprocessed:
            imgs = self.processor(images=images, return_tensors="pt", padding=True)
        else:
            imgs = {'pixel_values':images}
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
    
clip_transforms = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.Resize(size=224),
    transforms.CenterCrop(224),
    #transforms.ToPILImage(),
    transforms.ToTensor(),  # PIL -> ToTensor :: [0,255] -> [0.,1.]
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], 
                         [0.26862954, 0.26130258, 0.27577711]),
    
])

# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
# NB: might be able to use lightning datamodule to parallelize data processing
class RecusiveImagesPath(Dataset):
    def __init__(self, 
                 root='sample_data/images', 
                 transforms=clip_transforms,
                 #assign_ids=True, # doesn't need to be optional...
                 next_id=0
                ):
        if isinstance(root, str):
            root = Path(root)
        assert root.is_dir()
        self.root = root
        self.transforms = transforms
        #self.assign_ids = assign_ids
        self.next_id = next_id
        self.get_image_paths()
    def get_image_paths(self):
        self.img_paths = []
        for path_obj in self.root.glob('*'):
            if self._already_indexed(path_obj):
                continue
            if imghdr.what(path_obj) is not None:
                self.img_paths.append((path_obj, self.next_id))
                self.next_id += 1
    def _already_indexed(self, path_obj):
        # implement this later, assume all images are new for now
        return False
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        path_obj, im_id = self.img_paths[idx]
        path = str(path_obj)
        im = Image.open(path)
        if self.transforms is not None:
            im = self.transforms(im)
        return im, path, im_id



# fire cli?
# https://github.com/google/python-fire
class SearchyClient:
    def __init__(self,
                 images_root,
                 index_path='index.faissindex',
                 cfg=None):
        """
        images_root: root directory hosting images, will be recursively walked for indexing.
        index_path: path to faiss index
        """
        self.images_root = Path(images_root)
        self.index_path = Path(index_path)
        if cfg:
            self.from_config(cfg)
        self.clip = CLIP()
        self._get_vector_store() #create_FAISS()
        self._get_filepath_db() #create_sqlite() # might not be necessary
        
        
    def _get_vector_store(self):
        #pass
        if my_file.exists():
            self._vector_store = faiss.read_index(str(self.index_path))
        else:
            index = faiss.IndexFlatIP(self.clip.n_feats)
            self._vector_store = faiss.IndexIDMap(index)
    
    def _get_filepath_db(self):
        pass

    def from_config(self, cfg):
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

if __name__ == '__main__':
    dataset = RecusiveImagesPath()
    img_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    for i, (batch, paths) in enumerate(img_loader):
        continue
