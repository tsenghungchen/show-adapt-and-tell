#!/bin/bash
# download mscoco images
mkdir coco
cd coco
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip 
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/test2014.zip
unzip train2014.zip
unzip val2014.zip
unzip test2014.zip
rm train2014.zip
rm val2014.zip
rm test2014.zip
cd ..
# please download the pretrained ResNet-101 model at https://github.com/KaimingHe/deep-residual-networks
mkdir mscoco_data
# extract resnet feature and pack in pickle format
python extract_resnet_coco.py --def deep-residual-networks/prototxt/ResNet-101-deploy.prototxt --net resnet_model/ResNet-101-model.caffemodel --gpu 0
