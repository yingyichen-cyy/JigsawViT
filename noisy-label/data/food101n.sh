wget https://iudata.blob.core.windows.net/food101/Food-101N_release.zip

wget https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz

unzip Food-101N_release.zip
tar -xvf food-101.tar.gz

# change the data_path before preprocessing 
python3 preprocess_food101n.py --out-dir ./Food101N

