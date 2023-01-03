# We only use Nested+Co-teaching for finetuning
python3 main.py \
        --data-path ../data/Clothing1M/ \
        --data-set 'Clothing1M' \
        --arch 'vit_small_patch16' \
        --input-size 224 \
        --batch-size 96 \
        --max-lr 5e-5 \
        --min-lr 1e-5 \
        --total-iter 50000 \
        --warmup-iter 0 \
        --niter-eval 1000 \
        --eta 1 \
        --out-dir ./finetune_nested/Clothing1M_eta1_iter20k_wd005_mask025_smallP16_aug_bs96_warm0_fgr0.3_lr5e-5_acc_72.4_72.7_nested100 \
        --weight-decay 0.05 \
        --mask-ratio 0.75 \
        --nested 100 \
        --emb-dim 384 \
        --smoothing 0.1 \
        --mixup 0.8 \
        --cutmix 1.0 \
        --Gradual-iter 0 \
        --forgetRate 0.6 \
        --resumePthList ./checkpoints/Clothing1M_eta1_iter400k_wd005_mask025_smallP16_aug_warm20k_Acc0.724  ./checkpoints/Clothing1M_eta1_iter400k_wd005_mask025_smallP16_aug_warm20k_Acc0.727 \
        --gpu 0
