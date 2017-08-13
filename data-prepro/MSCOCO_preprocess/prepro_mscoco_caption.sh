# download and preprocess captions
wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip
unzip captions_train-val2014.zip
rm captions_train-val2014.zip
cd K_split
python preprocess_entity.py train
python preprocess_entity.py test
python preprocess_entity.py val
python preprocess_token.py train
python preprocess_token.py val
python preprocess_token.py test
mkdir -p ../CUB200_preprocess/cub_data
ln -s mscoco_data/dictionary_5.npz ../CUB200_preprocess/cub_data/
ln -s K_cleaned_words.npz ../CUB200_preprocess/ 
