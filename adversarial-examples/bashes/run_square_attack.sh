# black-box attack Square
# Square attack is a black-box attack, however, we write it in white_attack.py only for convenience.

python3 white_attack.py --attack-type Square --eps 0.063 --n-queries 100 --data-path ./data/imagenet/ --resumePth ./pretrained/imagenet-deit_small_patch16_224-org-acc78.85.pth --gpu 0

python3 white_attack.py --attack-type Square --eps 0.063 --n-queries 100 --data-path ./data/imagenet/ --resumePth ./pretrained/imagenet-deit_small_patch16_224-jigsaw-eta0.1-r0.5-acc80.51.pth --gpu 0
