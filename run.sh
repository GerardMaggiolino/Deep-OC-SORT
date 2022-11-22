# python3 main.py --exp_name $exp --post --new_kf_off --w_assoc_emb 3 --dataset mot17
# python3 main.py --exp_name $exp --post --new_kf_off --w_assoc_emb 3 --dataset mot20

exp=1122_final_test
# python3 main.py --exp_name $exp --post --new_kf_off --w_assoc_emb 3 --dataset mot17 --test_dataset
# python3 main.py --exp_name $exp --post --new_kf_off --w_assoc_emb 3 --dataset mot20 --test_dataset

exp=1122_final
python3 main.py --exp_name $exp --post --new_kf_off --w_assoc_emb 3 --dataset mot17
python3 main.py --exp_name $exp --post --new_kf_off --w_assoc_emb 3 --dataset mot20

#exp="1122_dance_emb_0.25"
#python3 main.py --exp_name $exp --post --cmc_off --new_kf_off --aw_off --w_assoc_emb 0.25 --dataset dance --aspect_ratio_thresh=10000000000
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


#exp="1114_kf"
#python3 main.py --exp_name $exp --post --cmc_off --aw_off --emb_off
#python3 main.py --exp_name $exp --post --cmc_off --aw_off --emb_off  --dataset mot20

#exp="1114_cmc_kf"
#python3 main.py --exp_name $exp --post --aw_off --emb_off
#python3 main.py --exp_name $exp --post --aw_off --emb_off  --dataset mot20

#exp="1114_cmc"
#python3 main.py --exp_name $exp --post --aw_off --emb_off --new_kf_off
#python3 main.py --exp_name $exp --post --aw_off --emb_off --new_kf_off --dataset mot20

#exp="1114_emb"
#python3 main.py --exp_name $exp --post --cmc_off --aw_off --new_kf_off
#python3 main.py --exp_name $exp --post --cmc_off --aw_off --new_kf_off --dataset mot20

#exp="1114_emb_aw"
#python3 main.py --exp_name $exp --post --cmc_off --new_kf_off
#python3 main.py --exp_name $exp --post --cmc_off --new_kf_off --dataset mot20

#exp="1114_emb_cmc"
#python3 main.py --exp_name $exp --post --aw_off --new_kf_off
#python3 main.py --exp_name $exp --post --aw_off --new_kf_off --dataset mot20

#exp="1114_emb_cmc_aw_bottom_0.5_w_assoc_0.5"
#python3 main.py --exp_name $exp --post --new_kf_off
#python3 main.py --exp_name $exp --post --new_kf_off --dataset mot20

# exp="1114_emb_aw_bottom_0.5_w_assoc_0.75"
# python3 main.py --exp_name $exp --post --cmc_off --new_kf_off --w_assoc_emb 0.75
# python3 main.py --exp_name $exp --post --cmc_off --new_kf_off --dataset mot20 --w_assoc_emb 0.75

#exp="1114_emb_cmc_aw_kf"
#python3 main.py --exp_name $exp --post
#python3 main.py --exp_name $exp --post --dataset mot20


