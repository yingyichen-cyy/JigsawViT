# Please change "--data-path" and "--out-dir" to your own paths
# 8 V100. each batch_size = 128
python -m torch.distributed.launch --nproc_per_node=8 --use_env main_jigsaw.py --model jigsaw_base_patch16_224 --batch-size 128 --data-path ./imagenet --lambda-jigsaw 0.1 --mask-ratio 0.5 --output_dir ./jigsaw_base_results
