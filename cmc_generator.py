import os
import glob
import json
import pdb

import cv2
import numpy as np


def main():
    dataset = NumpyMOTDataset(dataset="dance")
    tracker = KLTTracker()

    prev_img = None
    curr_seq = None
    rolling_A = np.eye(2, 3)

    for img, tag in dataset:
        seq = tag.split(":")[0]
        if curr_seq is not None and seq != curr_seq:
            tracker = KLTTracker()
            curr_seq = seq
            prev_img = None

        A = tracker.forward(img)
        if prev_img is None:
            curr_seq = seq
            prev_img = img
            continue
        rolling_A[:2, :2] = A[:2, :2].T @ rolling_A[:2, :2]
        rolling_A[:, 2] -= A[:, 2]

        #if np.linalg.norm(A[:, 2]) < 0.5:
        #    print(f"Skipping {tag}", end="\r")
        #    continue

        warped_img = cv2.warpAffine(img, rolling_A, (img.shape[1], img.shape[0]))

        cv2.imshow("stab", warped_img)
        cv2.waitKey(0)
        prev_img = img


class KLTTracker:
    def __init__(self):
        self.max_tracks = 3000
        self.min_points = 10
        self.track_length = 10
        self.forward_backward_thresh = 2
        self.corner_params = dict(
            maxCorners=self.max_tracks,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7,
            useHarrisDetector=False,
        )
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=3
        )

        self.prev_frame = None
        self.prev_mask = None
        self.tracks = []

    def forward(self, curr_img, boxes=None):
        A = np.eye(2, 3)
        curr_frame = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        mask = None
        if boxes is not None:
            mask = np.ones_like(curr_frame, dtype=np.uint8)
            bbox = np.round(boxes).astype(np.int32)
            bbox[bbox < 0] = 0
            for bb in bbox:
                mask[bb[1]:bb[3], bb[0]:bb[2]] = 0

        if self.prev_frame is None:
            self.prev_frame = curr_frame
            self.prev_mask = mask
            return A

        if len(self.tracks) < self.max_tracks:
            num_select = self.max_tracks - len(self.tracks)
            params = self.corner_params.copy()
            params["maxCorners"] = num_select
            kp = cv2.goodFeaturesToTrack(curr_frame, mask=mask, **params)
            kp = kp.reshape(-1, 2)
            for new_point in kp:
                self.tracks.append([new_point])

        p0 = np.array([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
        p1, status, err = cv2.calcOpticalFlowPyrLK(self.prev_frame, curr_frame, p0, None, **self.lk_params)
        p0_back, status, err = cv2.calcOpticalFlowPyrLK(curr_frame, self.prev_frame, p1, None, **self.lk_params)
        point_dist = np.sqrt(((p1 - p0_back).reshape(-1, 2) ** 2).sum(axis=-1))
        good_point = point_dist < self.forward_backward_thresh

        p1 = p1.reshape(-1, 2)
        continue_tracks = []
        for idx in range(len(self.tracks)):
            if not good_point[idx]:
                continue
            self.tracks[idx].append(p1[idx])
            if len(self.tracks[idx]) > self.track_length:
                del self.tracks[idx][0]
            continue_tracks.append(self.tracks[idx])
        self.tracks = continue_tracks

        # Find rigid matrix
        p0 = p0.reshape(-1, 2)[good_point]
        p1 = p1.reshape(-1, 2)[good_point]
        if p0.shape[0] > self.min_points:
            A, _ = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC)
        else:
            print("Warning: not enough matching points")

        self.prev_frame = curr_frame
        self.prev_mask = mask

        return A


class NumpyMOTDataset:
    # GT index
    POSITION_MAPPER = {
        "frame": 0,
        "identity": 1,
        "bb_left": 2,
        "bb_top": 3,
        "bb_width": 4,
        "bb_height": 5,
        "considered": 6,
        "class": 7,
        "visibilty": 8
    }
    SUPPORTED = {"dance", "mot17", "mot20"}

    def __init__(self, root_path="data", dataset="mot17", test=False, skip=()):
        self.image_names = []
        assert dataset in self.SUPPORTED, f"Loader paths only works for {self.SUPPORTED}"

        # Hardcoded paths
        if dataset == "mot17":
            direc = "mot"
            if test:
                name = "test"
                annotation = "test.json"
            else:
                name = "train"
                annotation = "val_half.json"
        elif dataset == "mot20":
            direc = "MOT20"
            if test:
                name = "test"
                annotation = "test.json"
            else:
                name = "train"
                annotation = "val_half.json"
        elif dataset == "dance":
            direc = "dancetrack"
            if test:
                name = "test"
                annotation = "test.json"
            else:
                annotation = "val.json"
                name = "val"
        else:
            raise RuntimeError("Specify path here.")

        path = os.path.join(root_path, direc)
        with open(os.path.join(path, "annotations", annotation), "r") as fp:
            self.annotations = json.load(fp)["images"]

        for info in self.annotations:
            img_path = os.path.join(path, name, info["file_name"])
            img_id = info["frame_id"]
            tag = info["file_name"].split("/")[0] + ":" + str(img_id)
            for s in skip:
                if s in tag:
                    break
            else:
                self.image_names.append((img_path, tag))

    def __getitem__(self, idx):
        """
        Returns numpy image and the image tag.

        """
        if idx >= len(self.image_names):
            raise IndexError(f"{idx} invalid, dataset length: {len(self.image_names)}")
        path, tag = self.image_names[idx]
        bgr_image = cv2.imread(path)

        return bgr_image, tag

    def __iter__(self):
        def dset_iterator():
            for idx in range(len(self)):
                yield self[idx]

        return dset_iterator()

    def __len__(self):
        return len(self.image_names)


if __name__ == "__main__":
    main()
