import pdb
import os

import torch
import cv2
import numpy as np

import dataset
import utils
from external.adaptors import yolox
from trackers import ocsort_embedding as tracker_module


def get_main_args():
    parser = tracker_module.args.make_parser()
    parser.add_argument("--result_folder", type=str, default="results/trackers/MOT17-val")
    parser.add_argument("--exp_name", type=str, default="exp1")
    parser.add_argument("--min_box_area", type=float, default=10, help="filter out tiny boxes")
    parser.add_argument(
        "--aspect_ratio_thresh",
        type=float,
        default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value.",
    )
    parser.add_argument("--post", type=bool, default=True, help="run post-processing linear interpolation.")
    return parser.parse_args()


def main():
    """Hardcoded to MOT17, OCSort for now."""
    # Set up loader, detector, tracker
    args = get_main_args()
    loader = dataset.get_mot17_loader()
    detector = yolox.get_model("external/weights/bytetrack_ablation.pth.tar")
    tracker = tracker_module.ocsort.OCSort(
        det_thresh=args.track_thresh,
        iou_threshold=args.iou_thresh,
        asso_func=args.asso,
        delta_t=args.deltat,
        inertia=args.inertia,
    )
    results = {}

    # See __getitem__ of dataset.MOTDataset
    for img, label, info, idx in loader:
        # Frame info
        img = img.cuda()
        frame_id = info[2].item()
        video_name = info[4][0].split("/")[0]
        if video_name not in results:
            results[video_name] = []
        # Initialize tracker on first frame of a new video
        print(f"Processing {video_name}:{frame_id}\r", end="")
        if frame_id == 1:
            print(f"Initializing tracker for {video_name}")
            tracker = tracker_module.ocsort.OCSort(
                det_thresh=args.track_thresh,
                iou_threshold=args.iou_thresh,
                asso_func=args.asso,
                delta_t=args.deltat,
                inertia=args.inertia,
            )

        # Nx5 of (x1, y1, x2, y2, conf)
        with torch.no_grad():
            pred = detector(img)

        # Nx5 of (x1, y1, x2, y2, ID)
        targets = tracker.update(pred, info, args.tsize, img)
        tlwhs, ids = utils.filter_targets(targets, args.aspect_ratio_thresh, args.min_box_area)
        results[video_name].append((frame_id, tlwhs, ids))

    # Save for all sequences
    folder = os.path.join(args.result_folder, args.exp_name, "data")
    os.makedirs(folder, exist_ok=True)
    for name, res in results.items():
        result_filename = os.path.join(folder, f"{name}.txt")
        utils.write_results_no_score(result_filename, res)
    print(f"Finished, results saved to {folder}")
    if args.post:
        utils.dti(folder, folder)
        print("Linear interpolation post-processing applied.")


def draw(name, pred, shape):
    name = os.path.join("data/mot/train", name)
    img = cv2.imread(name)
    img = cv2.resize(img, (shape[1], shape[0]))
    for p in pred:
        p = np.round(p).astype(np.int32)
        cv2.rectangle(img, (p[0], p[1]), (p[2], p[3]), (255, 0, 0), 5)
    cv2.imwrite("debug.png", img)


if __name__ == "__main__":
    main()
