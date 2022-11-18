# MOT Capstone

### TODOs 
- Move away from submodules. It's difficult to maintain paths and adaptors.
- Support DanceTrack 
- Streamline installation.

### Installation 

After cloning, run: 
`git submodule update --init` to pull external dependencies (detectors, benchmark evaluators).


Run `python setup.py develop` (possibly broken, but should include YOLOX setup.py).

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
```
and run:
```
python3 data/tools/convert_mot17_to_coco.py
python3 data/tools/convert_mot20_to_coco.py
```



### Evaluation

Set `exp=exp1`


For the baseline, MOT17/20: 
```
python3 main.py --exp_name $exp --post --emb_off --cmc_off --aw_off --new_kf_off --dataset mot17
python3 main.py --exp_name $exp --post --emb_off --cmc_off --aw_off --new_kf_off --dataset mot20
```

For the best results so far: 
```
python3 main.py --exp_name $exp --post --new_kf_off --w_assoc_emb 0.75
python3 main.py --exp_name $exp --post --new_kf_off --w_assoc_emb 0.75 --dataset mot20
```

This will create results at `results/trackers/MOT<17/20>-val/$exp`.

To run TrackEval for HOTA, on this new results, run: 
```bash
python3 external/TrackEval/scripts/run_mot_challenge.py \
  --SPLIT_TO_EVAL val \
  --METRICS HOTA \
  --TRACKERS_TO_EVAL $exp \
  --GT_FOLDER results/gt/ \
  --TRACKERS_FOLDER results/trackers/ \
  --BENCHMARK MOT17
```

### Contributing 
Formatted with `black --line-length=120 --exclude external .`



