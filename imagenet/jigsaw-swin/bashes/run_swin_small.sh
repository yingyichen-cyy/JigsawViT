# fine-tune a Swin-S model pre-trained on ImageNet-22K(21K):

# we have 8 gpus
python -m torch.distributed.launch \
	--nproc_per_node 8 \
	--master_port 10011 main.py \
	--cfg configs/swin/swin_small_patch4_window7_224_22kto1k_finetune.yaml \
	--pretrained ./pretrained/swin_small_patch4_window7_224_22k.pth \
	--data-path ./imagenet \
	--batch-size 128 \
	--accumulation-steps 2 \
	--use-checkpoint