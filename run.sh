# python3 main.py --exp_name $exp --post --new_kf_off --w_assoc_emb 3 --dataset mot17
# python3 main.py --exp_name $exp --post --new_kf_off --w_assoc_emb 3 --dataset mot20

# exp=1122_final_test
# python3 main.py --exp_name $exp --post --grid_off --new_kf_off --w_assoc_emb 3 --dataset mot17 --test_dataset
# python3 main.py --exp_name $exp --post --new_kf_off --w_assoc_emb 3 --dataset mot20 --test_dataset

# exp="1206_baseline"
# python3 main.py --exp_name $exp --post --emb_off --cmc_off --new_kf_off --aw_off --grid_off --dataset dance --aspect_ratio_thresh=10000000000
#exp="1122_dance_emb_0.5"
#python3 main.py --exp_name $exp --post --cmc_off --new_kf_off --aw_off --w_assoc_emb 0.5 --dataset dance --aspect_ratio_thresh=10000000000
#exp="1122_dance_emb_0.75"
#python3 main.py --exp_name $exp --post --cmc_off --new_kf_off --aw_off --w_assoc_emb 0.75 --dataset dance --aspect_ratio_thresh=10000000000
# exp="1122_dance_emb_1.5_fixed"
# python3 main.py --exp_name $exp --post --cmc_off --new_kf_off --aw_off --w_assoc_emb 1.5 --dataset dance --aspect_ratio_thresh=10000000000
# exp="1122_dance_aw_emb_4"
# python3 main.py --exp_name $exp --post --cmc_off --new_kf_off --w_assoc_emb 4 --dataset dance --aspect_ratio_thresh=10000000000
#exp="1122_dance_aw_emb_3"
#python3 main.py --exp_name $exp --post --cmc_off --new_kf_off --w_assoc_emb 3 --dataset dance --aspect_ratio_thresh=10000000000
#exp="1122_dance_baseline"
#python3 main.py --exp_name $exp --post --aw_off --emb_off --cmc_off --new_kf_off --dataset dance --aspect_ratio_thresh=10000000000


# exp=1205_baseline
# python3 main.py --exp_name $exp --post --grid_off --emb_off --aw_off --cmc_off --new_kf_off --w_assoc_emb 3 --dataset mot20

# python3 main.py --exp_name $exp --post --aw_off --cmc_off --new_kf_off --w_assoc_emb 3 --dataset mot17
# python3 main.py --exp_name $exp --post --aw_off --cmc_off --new_kf_off --w_assoc_emb 3 --dataset mot20
# exp=1206_de_cmc
# python3 main.py --exp_name $exp --post  --aw_off --grid_off --new_kf_off --dataset mot20

exp=0209_validation_baseline
python3 main.py --exp_name $exp --post --new_kf_off --grid_off --emb_off --aw_off --cmc_off --dataset mot17
exp=0209_validation_emb
python3 main.py --exp_name $exp --post --new_kf_off --grid_off --aw_off --cmc_off --dataset mot17
exp=0209_validation_cmc
python3 main.py --exp_name $exp --post --new_kf_off --grid_off --aw_off --emb_off --dataset mot17
exp=0209_validation_cmc_emb
python3 main.py --exp_name $exp --post --new_kf_off --grid_off --aw_off --dataset mot17
exp=0209_validation_cmc_emb_aw
python3 main.py --exp_name $exp --post --new_kf_off --grid_off --w_assoc_emb 3 --dataset mot17
exp=0209_validation_cmc_emb_aw_kf
python3 main.py --exp_name $exp --post --grid_off --w_assoc_emb 3 --dataset mot17



