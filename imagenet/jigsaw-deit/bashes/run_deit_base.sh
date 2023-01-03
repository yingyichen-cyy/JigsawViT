# Please change "--data-path" and "--out-dir" to your own paths
# 8 V100. each batch_size = 128
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_224 --batch-size 128 --data-path ./imagenet --output_dir ./deit_base_results