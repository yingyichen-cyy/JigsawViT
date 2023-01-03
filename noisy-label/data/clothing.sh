## You can also experiment on Clothing1M dataset
mkdir ./Clothing1M/
cd ./Clothing1M/

# Download Clothing1M
# You can download from https://drive.google.com/file/d/1XKw1xVwwsJWE9sFRE-dojw81sclUcVAi/view?usp=sharing
tar -xvf lukasmyth-datasets-clothing1m-4.tar

cd ..

# Generate two random Clothing1M noisy subsets for training
python3 clothing1M_rand_subset.py --name noisy_rand_subtrain --data-dir ./Clothing1M/ --seed 123