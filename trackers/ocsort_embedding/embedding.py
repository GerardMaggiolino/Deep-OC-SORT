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
        os.makedirs("./cache/embeddings/", exist_ok=True)
        self.cache_path = "./cache/embeddings/{}_embedding.pkl"
        self.cache = {}
        self.cache_name = ""

    def load_cache(self, path):
        self.cache_name = path
        cache_path = self.cache_path.format(path)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as fp:
                self.cache = pickle.load(fp)

    def get_horizontal_split_patches(image, bbox, patch_shape, img_info):

        bbox = np.array(bbox)

        bbox = bbox.astype(np.int)

        if (
            bbox[0] < 0
            or bbox[1] < 0
            or bbox[2] > image.shape[1]
            or bbox[3] > image.shape[0]
        ):

            print("FAULTY PATCH $$$ - ", bbox)
            bbox[0] = np.clip(bbox[0], 0, None)
            bbox[1] = np.clip(bbox[1], 0, None)
            bbox[2] = np.clip(bbox[2], 0, image.shape[1])
            bbox[3] = np.clip(bbox[3], 0, image.shape[0])
            print("FIXED PATCH $$$ - ", bbox)

        x1, y1, x2, y2 = bbox

        w = x2 - x1
        h = y2 - y1

        ### TODO - Write a generalized split logic

        split_boxes = [
            [x1, y1, x1 + w, y1 + h / 3],
            [x1, y1 + h / 3, x1 + w, y1 + (2 / 3) * h],
            [x1, y1 + (2 / 3) * h, x1 + w, y1 + h],
        ]

        split_boxes = np.array(split_boxes, dtype="int")

        patches = []
        for b in split_boxes:
            sx, sy, ex, ey = b
            im1 = image[sy:ey, sx:ex]

            # print("BOX ", b)
            # print("Image ", image.shape)

            dirs = "./VIZ/{}".format(img_info[4][0].split(".")[0])
            # Path(dirs).mkdir(parents=True, exist_ok=True)
            # cv2.imwrite(os.path.join(dirs, "{}.png".format(idx)), image)

            im1 = cv2.resize(im1, tuple(patch_shape[::-1]))
            patches.append(im1)

        patches = np.array(patches)
        print("ALL SPLIT PATCHES SHAPE - ", patches.shape)

        return patches

    def compute_embedding(self, img, bbox, tag):
        if self.cache_name != tag.split(":")[0]:
            self.load_cache(tag.split(":")[0])

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
        model = torchreid.models.build_model(
            name="osnet_ain_x1_0", num_classes=2510, loss="softmax", pretrained=False
        )
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
        if self.cache_name:
            with open(self.cache_path.format(self.cache_name), "wb") as fp:
                pickle.dump(self.cache, fp)
