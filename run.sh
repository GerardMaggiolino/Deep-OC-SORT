#exp="1114_baseline"
#python3 main.py --exp_name $exp --post --emb_off --cmc_off --aw_off --new_kf_off
#python3 main.py --exp_name $exp --post --emb_off --cmc_off --aw_off --new_kf_off --dataset mot20
#python3 main.py --exp_name $exp --post --emb_off --cmc_off --aw_off --new_kf_off --dataset dance

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

exp="1114_emb_aw_bottom_0.5_w_assoc_0.75"
python3 main.py --exp_name $exp --post --cmc_off --new_kf_off --w_assoc_emb 0.75
python3 main.py --exp_name $exp --post --cmc_off --new_kf_off --dataset mot20 --w_assoc_emb 0.75

#exp="1114_emb_cmc_aw_kf"
#python3 main.py --exp_name $exp --post
#python3 main.py --exp_name $exp --post --dataset mot20

# To test: give scale to CMC
# To test: turn off velocity for scale when propogating
