exp=$1
python main.py --exp_name $exp
python3 external/TrackEval/scripts/run_mot_challenge.py \
  --SPLIT_TO_EVAL val \
  --METRICS HOTA CLEAR Identity \
  --TRACKERS_TO_EVAL $exp \
  --GT_FOLDER results/gt/ \
  --TRACKERS_FOLDER results/trackers/ | tail -n 26 | head -n 8 | awk '{print $1 "\t" $2}'