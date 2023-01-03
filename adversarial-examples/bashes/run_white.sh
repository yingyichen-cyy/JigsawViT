# white-box attack

python3 white_attack.py --attack-type FGSM --eps 0.125 --data-path ./data/imagenet/ --resumePth ./pretrained/imagenet-deit_small_patch16_224-org-acc78.85.pth --gpu 0

python3 white_attack.py --attack-type FGSM --eps 0.125 --data-path ./data/imagenet/ --resumePth ./pretrained/imagenet-deit_small_patch16_224-jigsaw-eta0.1-r0.5-acc80.51.pth --gpu 0


# white-box attack PGD

python3 white_attack.py --attack-type PGD --arch vit --steps 5 --eps 0.125 --data-path ./data/imagenet/ --resumePth ./pretrained/imagenet-deit_small_patch16_224-org-acc78.85.pth --gpu 0

python3 white_attack.py --attack-type PGD --arch vit --steps 5 --eps 0.125 --data-path ./data/imagenet/ --resumePth ./pretrained/imagenet-deit_small_patch16_224-jigsaw-eta0.1-r0.5-acc80.51.pth --gpu 0


python3 white_attack.py --attack-type PGD --arch vit --steps 7 --eps 0.031 --data-path ./data/imagenet/ --resumePth ./pretrained/imagenet-deit_small_patch16_224-org-acc78.85.pth --gpu 0

python3 white_attack.py --attack-type PGD --arch vit --steps 7 --eps 0.031 --data-path ./data/imagenet/ --resumePth ./pretrained/imagenet-deit_small_patch16_224-jigsaw-eta0.1-r0.5-acc80.51.pth --gpu 0


# white attack CW

python3 white_attack.py --attack-type CW --arch vit --steps 10 --data-path ./data/imagenet/ --resumePth ./pretrained/imagenet-deit_small_patch16_224-org-acc78.85.pth --gpu 0

python3 white_attack.py --attack-type CW --arch vit --steps 10 --data-path ./data/imagenet/ --resumePth ./pretrained/imagenet-deit_small_patch16_224-jigsaw-eta0.1-r0.5-acc80.51.pth --gpu 0

