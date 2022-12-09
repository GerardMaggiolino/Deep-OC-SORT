import os
import glob
import json
import pickle
import pdb

import cv2
import numpy as np


def main():
    with open("cache/det_bytetrack_ablation.pkl", "rb") as fp:
        detections = pickle.load(fp)

    output_path = "./cache/cmc_files/MOT17_custom_ablation/GMC-{}.txt"
    dataset = NumpyMOTDataset(dataset="mot17")
    tracker = KLTTracker()
    scorer = Evaluator()

    curr_seq = None
    data = []

    for img, tag in dataset:
        print(f"{tag}\r", end="")
        seq = tag.split(":")[0]
        if curr_seq is not None and seq != curr_seq:
            print(f"Writing for {curr_seq}.")
            print(f"Skipped {tracker.skipped_frames}/{len(data)} frames.")
            before, after = scorer.get_scores()
            print(f"Mean L1 distance before ({before:.2f}) and after ({after:.2f}) correction")
            write_cmc(output_path.format(curr_seq), data)
            tracker = KLTTracker()
            scorer = Evaluator()

        scale = min(800 / img.shape[0], 1440 / img.shape[1])
        bbox = detections[tag].numpy()
        bbox = bbox[bbox[:, 4] > 0.5][:, :4] / scale

        A = tracker.forward(img, bbox)
        scorer.forward(img, A)
        data.append(A)
        curr_seq = seq


def write_cmc(path, data):
    with open(path, "w") as fp:
        for idx, A in enumerate(data):
            line = "\t".join([f"{s:.10f}" for s in ([idx] + A.reshape(-1).tolist())]) + "\n"
            fp.write(line)


class Evaluator:
    def __init__(self):
        self.prior_scores = 0
        self.new_scores = 0
        self.count = 0
        self.img_scale = 1
        self.every = 5

        self.prev_img = None
        self.transform = np.eye(3, 3)

    def forward(self, img, transform):
        img = cv2.resize(img, None, fx=self.img_scale, fy=self.img_scale)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        transform[:, 2] *= self.img_scale

        if self.prev_img is None:
            self.prev_img = img
            return

        self.count += 1
        # Compose transform and evaluate every few frames
        transform = np.concatenate((transform, np.array([[0, 0, 1]])), axis=0)
        self.transform = transform @ self.transform
        if self.count % self.every != 0:
            return

        mask = np.ones_like(self.prev_img)
        out = cv2.warpAffine(self.prev_img, self.transform[:2], (img.shape[1], img.shape[0]))
        out_mask = cv2.warpAffine(mask, self.transform[:2], (img.shape[1], img.shape[0])).astype(np.bool_)

        new_score = np.abs(out[out_mask] - img[out_mask]).mean()
        prior_score = np.abs(self.prev_img[out_mask] - img[out_mask]).mean()

        self.prior_scores += (prior_score - self.prior_scores) / self.count
        self.new_scores += (new_score - self.new_scores) / self.count
        if new_score > prior_score * 2:
            print("Very bad transform, defaulting to identity")
        self.prev_img = img
        self.transform = np.eye(3, 3)

    def get_scores(self):
        return self.prior_scores, self.new_scores


class KLTTracker:
    def __init__(self):
        self.max_tracks = 4000
        self.min_points = 10
        self.track_length = 10
        self.forward_backward_thresh = 1
        self.corner_params = dict(
            maxCorners=self.max_tracks,
            qualityLevel=0.01,
            useHarrisDetector=False,
        )
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=3
        )

        self.prev_frame = None
        self.prev_mask = None
        self.tracks = []
        self.skipped_frames = 0

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
            self.skipped_frames += 1

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
