import pickle
import os

import cv2
import numpy as np


class CMCComputer:
    def __init__(self):
        os.makedirs("./cache", exist_ok=True)
        self.cache_path = "./cache/affine_ocsort.pkl"
        self.cache = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as fp:
                self.cache = pickle.load(fp)

        self.prev_img = None
        self.prev_desc = None

    def compute_affine(self, img, bbox, tag):
        A = np.zeros((2, 3))
        A[:2, :2] = np.eye(2)
        return A

        if tag in self.cache:
            A = self.cache[tag]
            return A

        img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img *= np.array((0.229, 0.224, 0.225))
        img += np.array((0.485, 0.456, 0.406))
        img *= 255
        img = np.ascontiguousarray(img.clip(0, 255).astype(np.uint8))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_LINEAR)

        mask = np.ones_like(img, dtype=np.uint8)
        bbox = np.round(bbox).astype(np.int32)
        bbox[bbox < 0] = 0
        for bb in bbox:
            mask[bb[1]:bb[3], bb[0]:bb[2]] = 0

        detector = cv2.SIFT_create()
        kp, desc = detector.detectAndCompute(img, mask)
        if self.prev_desc is None:
            self.prev_desc = [kp, desc]
            return A

        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(self.prev_desc[1], desc, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > 10:
            src_pts = np.float32([self.prev_desc[0][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            A, _ = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5)
        if A is None:
            A = np.zeros((2, 3))
            A[:2, :2] = np.eye(2)

        if tag not in self.cache:
            self.cache[tag] = A
        self.prev_desc = [kp, desc]

        A[:, :2] = np.eye(2)

        return A

    def dump_cache(self):
        with open(self.cache_path, "wb") as fp:
            pickle.dump(self.cache, fp)
