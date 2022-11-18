import pdb
from collections import OrderedDict
import os
import pickle

import torch
import torchvision
import torchreid
import numpy as np

from external.adaptors.fastreid_adaptor import FastReID


class EmbeddingComputer:
    def __init__(self):
        self.model = None
        self.crop_size = (256, 128)
        os.makedirs("./cache", exist_ok=True)
        self.cache_path = "./cache/embedding_ocsort.pkl"
        self.cache = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as fp:
                self.cache = pickle.load(fp)

    def compute_embedding(self, img, bbox, tag):
        if tag in self.cache:
            embs = self.cache[tag]
            if embs.shape[0] != bbox.shape[0]:
                raise RuntimeError(
                    "ERROR: The number of cached embeddings don't match the "
                    "number of detections.\nWas the detector model changed? Delete cache if so."
                )
            return embs

        if self.model is None:
            self.initialize_model()

        # Make sure bbox is within image frame
        results = np.round(bbox).astype(np.int32)
        results[:, 0] = results[:, 0].clip(0, img.shape[3])
        results[:, 1] = results[:, 1].clip(0, img.shape[2])
        results[:, 2] = results[:, 2].clip(0, img.shape[3])
        results[:, 3] = results[:, 3].clip(0, img.shape[2])
        # Generate all the crops
        crops = []
        for p in results:
            crop = img[:, :, p[1] : p[3], p[0] : p[2]]
            try:
                crop = torchvision.transforms.functional.resize(crop, self.crop_size)
                crops.append(crop)
            except:
                print("Error generating crop for EmbeddingComputer")
                crops.append(torch.randn(3, *self.crop_size).cuda())
        crops = torch.cat(crops, dim=0)

        # Create embeddings and l2 normalize them
        with torch.no_grad():
            embs = self.model(crops)
        embs = torch.nn.functional.normalize(embs)
        embs = embs.cpu().numpy()

        self.cache[tag] = embs
        return embs

    def initialize_model(self):
        model = torchreid.models.build_model(name="osnet_ain_x1_0", num_classes=2510, loss="softmax", pretrained=False)
        sd = torch.load("external/weights/osnet_ain_ms_d_c.pth.tar")["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in sd.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
        model.eval()
        model.cuda()
        self.model = model

    def dump_cache(self):
        with open(self.cache_path, "wb") as fp:
            pickle.dump(self.cache, fp)
