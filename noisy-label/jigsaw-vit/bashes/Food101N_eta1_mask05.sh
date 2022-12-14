# Please change "--data-path" and "--out-dir" to your own paths
python3 main.py \
       --arch 'vit_small_patch16' \
       --input-size 224 \
       --batch-size 128 \
       --max-lr 1e-3 \
       --min-lr 1e-6 \
       --total-iter 400000 \
       --warmup-iter 20000 \
       --niter-eval 10000 \
       --weight-decay 0.05 \
       --smoothing 0.1 \
       --mixup 0.8 \
       --cutmix 1.0 \
       --dist-url 'tcp://localhost:10003' \
       --multiprocessing-distributed \
       --world-size 1 \
       --workers 16 \
       --rank 0 \
       --data-path ../data/Food101N \
       --data-set 'Food101N' \
       --eta 1.0 \
       --mask-ratio 0.5 \
       --out-dir ./LabelNoise_checkpoint/Food101N_eta1_mask05
