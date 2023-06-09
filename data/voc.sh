#!/bin/bash

# download PASCAL VOC 2007 dataset
#
# ref: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html

echo "Downloading..."

mkdir voc
cd voc

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar 
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar

tar -xvf VOCtrainval_06-Nov-2007.tar -C .
tar -xvf VOCtest_06-Nov-2007.tar -C .
tar -xvf VOCdevkit_08-Jun-2007.tar -C .

rm VOCtrainval_06-Nov-2007.tar
rm VOCtest_06-Nov-2007.tar
rm VOCdevkit_08-Jun-2007.tar

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar -C .

rm VOCtrainval_11-May-2012.tar


echo "Done."
