import pdb
import pickle
import os

import cv2
import numpy as np


class CMCComputer:
    def __init__(self, minimum_features=10, method="sparse"):
        assert method in ["sparse", "sift"]

        os.makedirs("./cache", exist_ok=True)
        self.cache_path = "./cache/affine_ocsort.pkl"
        self.cache = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as fp:
                self.cache = pickle.load(fp)
        self.minimum_features = minimum_features
        self.prev_img = None
        self.prev_desc = None
        self.sparse_flow_param = dict(
            maxCorners=1000,
            qualityLevel=0.01,
            minDistance=1,
            blockSize=3,
            useHarrisDetector=False,
            k=0.04,
        )

        self.comp_function = None
        if method == "sparse":
            self.comp_function = self._affine_sparse_flow
        elif method == "sift":
            self.comp_function = self._affine_sift

    def compute_affine(self, img, bbox, tag):
        img = cv2.cvtColor(img[0].numpy(), cv2.COLOR_BGR2GRAY)
        if tag in self.cache:
            A = self.cache[tag]
            return A
        print("CMC: not using cached")

        mask = np.ones_like(img, dtype=np.uint8)
        bbox = np.round(bbox).astype(np.int32)
        bbox[bbox < 0] = 0
        for bb in bbox:
            mask[bb[1] : bb[3], bb[0] : bb[2]] = 0

        A = self.comp_function(img, mask)
        if tag not in self.cache:
            self.cache[tag] = A

        return A

    def _affine_sift(self, frame, mask):
        A = np.eye(2, 3)
        detector = cv2.SIFT_create()
        kp, desc = detector.detectAndCompute(frame, mask)
        if self.prev_desc is None:
            self.prev_desc = [kp, desc]
            return A
        if desc.shape[0] < self.minimum_features or self.prev_desc[1].shape[0] < self.minimum_features:
            return A

        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(self.prev_desc[1], desc, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > self.minimum_features:
            src_pts = np.float32([self.prev_desc[0][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            A, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
        else:
            print("Warning: not enough matching points")
        if A is None:
            A = np.eye(2, 3)

        self.prev_desc = [kp, desc]
        return A

    def _affine_sparse_flow(self, frame, mask):
        # Initialize
        A = np.eye(2, 3)

        # find the keypoints
        keypoints = cv2.goodFeaturesToTrack(frame, mask=mask, **self.sparse_flow_param)

        # Handle first frame
        if self.prev_img is None:
            self.prev_img = frame
            self.prev_desc = keypoints
            return A

        matched_kp, status, err = cv2.calcOpticalFlowPyrLK(self.prev_img, frame, self.prev_desc, None)
        prev_points = self.prev_desc[status]
        curr_points = matched_kp[status]

        # Find rigid matrix
        if prev_points.shape[0] > self.minimum_features:
            A, _ = cv2.estimateAffinePartial2D(prev_points, curr_points, method=cv2.RANSAC)
        else:
            print("Warning: not enough matching points")
        if A is None:
            A = np.eye(2, 3)

        self.prev_img = frame
        self.prev_desc = keypoints
        return A

    def dump_cache(self):
        with open(self.cache_path, "wb") as fp:
            pickle.dump(self.cache, fp)
