#!/bin/bash

path=$1

cd path


wget http://visualqa.org/data/mscoco/vqa/Annotations_Train_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/Annotations_Val_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/v2_Annotations_Train_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/v2_Annotations_Val_mscoco.zip

wget http://visualqa.org/data/mscoco/vqa/Questions_Train_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/Questions_Test_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/Questions_Val_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Test_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Val_mscoco.zip

wget http://visualqa.org/data/abstract_v002/vqa/Annotations_Train_abstract_v002.zip
wget http://visualqa.org/data/abstract_v002/vqa/Questions_Train_abstract_v002.zip
wget http://visualqa.org/data/abstract_v002/vqa/Questions_Test_abstract_v002.zip
wget http://visualqa.org/data/abstract_v002/vqa/Annotations_Val_abstract_v002.zip
wget http://visualqa.org/data/abstract_v002/vqa/Questions_Val_abstract_v002.zip


unzip Annotations_Train_mscoco.zip
unzip Annotations_Val_mscoco.zip
unzip v2_Annotations_Train_mscoco.zip
unzip v2_Annotations_Val_mscoco.zip
unzip Annotations_Val_abstract_v002.zip
unzip Annotations_Train_abstract_v002.zip


unzip Questions_Train_mscoco.zip
unzip Questions_Val_mscoco.zip
unzip Questions_Test_mscoco.zip

unzip v2_Questions_Train_mscoco.zip
unzip v2_Questions_Val_mscoco.zip
unzip v2_Questions_Test_mscoco.zip

unzip Questions_Train_abstract_v002.zip
unzip Questions_Val_abstract_v002.zip
unzip Questions_Test_abstract_v002.zip

mkdir zip
mv *.zip zip

rm MultipleChoice*





ln -s mscoco_train2014_annotations.json vqa_train2014_annotations.json
ln -s mscoco_val2014_annotations.json vqa_val2014_annotations.json

ln -s v2_mscoco_train2014_annotations.json vqa_train2017_annotations.json
ln -s v2_mscoco_val2014_annotations.json vqa_val2017_annotations.json

ln -s abstract_v002_train2015_annotations.json vqa_train2015_annotations.json
ln -s abstract_v002_val2015_annotations.json vqa_val2015_annotations.json

ln -s OpenEnded_mscoco_train2014_questions.json vqa_train2014_questions.json
ln -s OpenEnded_mscoco_val2014_questions.json vqa_val2014_questions.json
ln -s OpenEnded_mscoco_test-dev2015_questions.json vqa_test-dev2014_questions.json
ln -s OpenEnded_mscoco_test2015_questions.json vqa_test2014_questions.json

ln -s v2_OpenEnded_mscoco_train2014_questions.json vqa_train2017_questions.json
ln -s v2_OpenEnded_mscoco_val2014_questions.json vqa_val2017_questions.json
ln -s v2_OpenEnded_mscoco_test2015_questions.json vqa_test2017_questions.json
ln -s v2_OpenEnded_mscoco_test-dev2015_questions.json vqa_test-dev2017_questions.json

ln -s OpenEnded_abstract_v002_val2015_questions.json vqa_val2015_questions.json
ln -s OpenEnded_abstract_v002_train2015_questions.json vqa_train2015_questions.json


mkdir images
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
wget http://msvocds.blob.core.windows.net/coco2015/test2015.zip

unzip train2014.zip -d images
unzip val2014.zip -d images
unzip test2015.zip -d images

ln -s images/test2015 images/test2014
ln -s images/test2015 images/test-dev2014
ln -s images/test2015 images/test2017
ln -s images/test2015 images/test-dev2017

ln -s images/train2014 images/train2017
ln -s images/val2014 images/val2017