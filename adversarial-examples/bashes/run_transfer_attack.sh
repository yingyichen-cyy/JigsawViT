# black-box attack
# Surrogate ViT
python3 transfer_attack.py --arch vit --attack-type FGSM --data-path ./data/imagenet/ --resumePth1 ./pretrained/imagenet-deit_small_patch16_224-org-acc78.85.pth --resumePth2 ./pretrained/imagenet-deit_small_patch16_224-jigsaw-eta0.1-r0.5-acc80.51.pth --gpu 0

python3 transfer_attack.py --arch vit --attack-type BIM --data-path ./data/imagenet/ --resumePth1 ./pretrained/imagenet-deit_small_patch16_224-org-acc78.85.pth --resumePth2 ./pretrained/imagenet-deit_small_patch16_224-jigsaw-eta0.1-r0.5-acc80.51.pth --gpu 0

python3 transfer_attack.py --arch vit --attack-type PGD --data-path ./data/imagenet/ --resumePth1 ./pretrained/imagenet-deit_small_patch16_224-org-acc78.85.pth --resumePth2 ./pretrained/imagenet-deit_small_patch16_224-jigsaw-eta0.1-r0.5-acc80.51.pth --gpu 0

python3 transfer_attack.py --arch vit --attack-type MI --data-path ./data/imagenet/ --resumePth1 ./pretrained/imagenet-deit_small_patch16_224-org-acc78.85.pth --resumePth2 ./pretrained/imagenet-deit_small_patch16_224-jigsaw-eta0.1-r0.5-acc80.51.pth --gpu 0

python3 transfer_attack.py --arch vit --attack-type AutoAttack --data-path ./data/imagenet/ --resumePth1 ./pretrained/imagenet-deit_small_patch16_224-org-acc78.85.pth --resumePth2 ./pretrained/imagenet-deit_small_patch16_224-jigsaw-eta0.1-r0.5-acc80.51.pth --gpu 0
