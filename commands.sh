CUDA_VISIBLE_DEVICES=1 python main.py --config configs/microtubule_mid_future1_sample1.yaml --dataset microtubule_mid
CUDA_VISIBLE_DEVICES=1 python main.py --config configs/receptor_mid_future1_sample1.yaml --dataset receptor_mid
CUDA_VISIBLE_DEVICES=1 python main.py --config configs/vesicle_mid_future1_sample1.yaml --dataset vesicle_mid

CUDA_VISIBLE_DEVICES=1 python main.py --config configs/microtubule_high_future1_sample1.yaml --dataset microtubule_high
# CUDA_VISIBLE_DEVICES=0 python main.py --config configs/microtubule_all_future1_sample1.yaml --dataset microtubule_all

CUDA_VISIBLE_DEVICES=1 python main.py --config configs/receptor_high_future1_sample1.yaml --dataset receptor_high
# CUDA_VISIBLE_DEVICES=2 python main.py --config configs/receptor_all_future1_sample1.yaml --dataset receptor_all

CUDA_VISIBLE_DEVICES=1 python main.py --config configs/vesicle_high_future1_sample1.yaml --dataset vesicle_high
# CUDA_VISIBLE_DEVICES=3 python main.py --config configs/vesicle_all_future1_sample1.yaml --dataset vesicle_all


