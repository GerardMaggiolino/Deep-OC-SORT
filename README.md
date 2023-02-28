# Deep-OC-SORT

[ArXiv submission.](https://arxiv.org/abs/2302.11813)

### NOTE FROM THE AUTHOR 

The repo isn't in a clean state at the moment, and the installation below could be broken. However, I'm now 
working full time at Tesla Autopilot and don't have a ton of free time to make QOL changes here. I'll update this 
ASAP.

Please try to install the YOLOX model and requirements. Below are the basic commands to run the validation datasets 
and verify the results in the paper. If something is broken after you've installed requirements, raise an issue. If you want to 
help, some order of importance tasks are: 

- *Streamlining the installation process, removing submodules and just going to flat folders.*
- Making the Kalman filter significantly less ugly and batched. 
- Removing unused functions, excess code.  

### Installation

After cloning, run:
`git submodule update --init` to pull external dependencies (detectors, benchmark evaluators).

*NOTE:* We'll move away from submodules ASAP. Having the folders here for you should be easier.  

Follow YOLOX installation instructions in its subdirectory.

Add [the weights](https://drive.google.com/file/d/1iqhM-6V_r1FpOlOzrdP_Ejshgk0DxOob/view) to the `external/weights` directory.

Then, install requirements with Python 3.8 and `pip install -r requirements.txt`.

### Data

Place MOT17/20 under:

```
data
|——————mot (this is MOT17)
|        └——————train
|        └——————test
|——————MOT20
|        └——————train
|        └——————test
|——————dancetrack
|        └——————train
|        └——————test
|        └——————val
```

and run:

```
python3 data/tools/convert_mot17_to_coco.py
python3 data/tools/convert_mot20_to_coco.py
python3 data/tools/convert_dance_to_coco.py
```

### Evaluation

Set `exp=baseline`

For the baseline, MOT17/20:

```
# Flags to disable all the new changes
python3 main.py --exp_name $exp --post --emb_off --grid_off --cmc_off --aw_off --new_kf_off --dataset mot17
python3 main.py --exp_name $exp --post --emb_off --grid_off --cmc_off --aw_off --new_kf_off --dataset mot20 --track_thresh 0.4
python3 main.py --exp_name $exp --post --emb_off --grid_off --cmc_off --aw_off --new_kf_off --dataset dance --aspect_ratio_thresh 1000
```

This will create results at:

```
# For the standard results
results/trackers/<DATASET NAME>-val/$exp.
# For the results with linear interpolation
results/trackers/<DATASET NAME>-val/${exp}_post.
```

To run TrackEval for HOTA with linear post processing MOT17, run:

```bash
python3 external/TrackEval/scripts/run_mot_challenge.py \
  --SPLIT_TO_EVAL val \
  --METRICS HOTA Identity \
  --TRACKERS_TO_EVAL ${exp}_post \
  --GT_FOLDER results/gt/ \
  --TRACKERS_FOLDER results/trackers/ \
  --BENCHMARK DANCE
```

Replace that last argument with MOT17 or MOT20 to evaluate those datasets.  

For the highest reported ablation results, run: 
```
exp=best_paper_ablations
python3 main.py --exp_name $exp --post --grid_off --new_kf_off --dataset mot17 --w_assoc_emb 0.75 --aw_param 0.5
python3 main.py --exp_name $exp --post --grid_off --new_kf_off --dataset mot20 --track_thresh 0.4 --w_assoc_emb 0.75 --aw_param 0.5
python3 main.py --exp_name $exp --post --grid_off --new_kf_off --dataset dance --aspect_ratio_thresh 1000 --w_assoc_emb 1.25 --aw_param 1
```

and re-run the TrackEval script given above. 

You can achieve higher results on individual datasets with different parameters, but we kept them fairly consistent with round 
numbers to avoid over-tuning.

### Contributing

Formatted with `black --line-length=120 --exclude external .`

# Citation

If you find our work useful, please cite our paper: 
```
@article{maggiolino2023deep,
    title={Deep OC-SORT: Multi-Pedestrian Tracking by Adaptive Re-Identification}, 
    author={Maggiolino, Gerard and Ahmad, Adnan and Cao, Jinkun and Kitani, Kris},
    journal={arXiv preprint arXiv:2302.11813},
    year={2023},
}
```

Also see OC-SORT, which we base our work upon: 
```
@article{cao2022observation,
  title={Observation-centric sort: Rethinking sort for robust multi-object tracking},
  author={Cao, Jinkun and Weng, Xinshuo and Khirodkar, Rawal and Pang, Jiangmiao and Kitani, Kris},
  journal={arXiv preprint arXiv:2203.14360},
  year={2022}
}
```
