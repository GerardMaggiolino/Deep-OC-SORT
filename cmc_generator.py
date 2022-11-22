import os
import glob
import json
import pdb

import cv2
import numpy as np


def main():
    dataset = NumpyMOTDataset(dataset="mot17")

    pdb.set_trace()


class KLTTracker:
    def __init__(self):
        self.max_tracks = 1000
        self.min_points = 10
        self.track_length = 15
        self.forward_backward_thresh = 1
        self.corner_params = dict(
            maxCorners=self.max_tracks,
            qualityLevel=0.1,
            minDistance=7,
            blockSize=7,
            useHarrisDetector=False,
        )
        self.lk_params = dict(
            win_size=(15, 15),
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
            kp = cv2.goodFeaturesToTrack(curr_frame, mask=mask, **self.corner_params)
            kp = kp.reshape(-1, 2)
            num_select = self.max_tracks - len(self.tracks)
            inds = [np.random.choice(kp.shape[0], replace=False, size=min(num_select, kp.shape[0]))]
            for new_point in kp[inds]:
                self.tracks.append([new_point])

        p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
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



class RankConstrainedBG:
    def __init__(self):
        self.buffer = []

    def apply(self, flow):
        pass

    def init_buffer(self, flows: list, gt=None):
        self.buffer = flows
        h, w, _ = flows[0].shape
        edge = 5
        x_cord, y_cord = np.meshgrid(
            np.arange(edge, w - edge), np.arange(edge, h - edge)
        )
        cords = np.stack((y_cord, x_cord), axis=2).astype(np.float32)
        mat = np.empty(
            (2 * (len(flows) + 1), (h - edge * 2) * (w - edge * 2)), np.float32
        )
        mat[:2] = cords.reshape(-1, 2).T

        for idx in range(len(flows)):
            func = interpolate.RegularGridInterpolator(
                points=(np.arange(h), np.arange(w)),
                values=flows[idx][:, :, ::-1],
                bounds_error=False,
                fill_value=0,
            )
            values = func(cords.reshape(-1, 2))
            values = values.reshape(h - edge * 2, w - edge * 2, -1)
            cords += values
            mat[(idx + 1) * 2 : (idx + 2) * 2] = cords.reshape(-1, 2).T

        if gt is not None:
            self.visualize_traj(mat, gt, flows)

        cords = cords.reshape(-1, 2)
        invalid = np.logical_or(np.any((cords < edge), axis=1), cords[:, 0] > h - edge)
        invalid = np.logical_or(invalid, cords[:, 1] > w - edge)
        traj_mask = self.ransac(mat)
        traj_mask = np.logical_or(traj_mask, invalid)

        in_points = mat[:, traj_mask][:2].astype(np.int32)
        mask = np.ones((h, w), dtype=np.bool_)
        mask[in_points[0], in_points[1]] = 0
        mask[:edge] = 0
        mask[-edge:] = 0
        mask[:, :edge] = 0
        mask[:, -edge:] = 0
        return mask

    def ransac(self, mat):
        best_num_inlier = -1
        best_inlier_mask = None

        for _ in range(50):
            select = np.random.choice(mat.shape[1], replace=False, size=3)
            W = mat[:, select]
            P = W.dot(np.linalg.inv(W.T.dot(W))).dot(W.T)
            error = np.linalg.norm(P.dot(mat) - mat, axis=0)

            inlier_mask = error < 4
            num_inlier = inlier_mask.sum()
            if num_inlier > best_num_inlier:
                best_num_inlier = num_inlier
                best_inlier_mask = inlier_mask

        return best_inlier_mask

    def visualize_traj(self, mat, pics, flows):
        samp = 300
        imgs = []
        traj = mat[:, np.random.choice(mat.shape[1], replace=False, size=samp)]

        p1 = pics[0].copy()
        p2 = flow_to_bgr(flows[0])
        for col in range(samp):
            points = traj[:, col].reshape(-1, 2)
            o = [round(points[0][1]), round(points[0][0])]
            cv2.circle(p1, o, 2, (0, 255, 0))
            cv2.circle(p2, o, 2, (0, 255, 0))
        imgs.append((p1, p2))

        for idx in range(len(pics) - 1):
            pic = pics[idx]
            flow = flow_to_bgr(flows[idx])
            cv2.imshow("original", pic)
            for col in range(samp):
                points = traj[:, col].reshape(-1, 2)
                o = [round(points[0][1]), round(points[0][0])]
                cv2.circle(pic, o, 2, (0, 255, 0))
                cv2.circle(flow, o, 2, (0, 255, 0))
                for idj in range(idx + 1):
                    p1 = [round(points[idj][1]), round(points[idj][0])]
                    p2 = [round(points[idj + 1][1]), round(points[idj + 1][0])]
                    cv2.line(pic, p1, p2, (255, 0, 0), 2)
                    cv2.line(flow, p1, p2, (255, 0, 0), 2)
            imgs.append((pic, flow))

        for idx in range(len(imgs)):
            imgs[idx] = np.concatenate(imgs[idx], axis=0)
        write_video(imgs, fps=1)


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

    def __init__(self, root_path="data", dataset="mot17", test=False):
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

            self.image_names.append((img_path, tag))

    def __getitem__(self, idx):
        """
        Returns numpy image, the image path, and the image tag.

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
