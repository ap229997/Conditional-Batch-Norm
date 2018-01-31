## Conditional Batch Normalization
Pytorch implementation of NIPS 2017 paper "Modulating early visual processing by language" 
[[Link]](https://papers.nips.cc/paper/7237-modulating-early-visual-processing-by-language.pdf) </br>

### Introduction
The authors present a novel approach to incorporate language information into extracting visual features by conditioning the Batch Normalization parameters on the language. They apply Conditional Batch Normalization (CBN) to a pre-trained ResNet and show that this significantly improves performance on visual question answering tasks. </br>

### Setup
This repository is compatible with python 2. </br>
- Follow instructions outlined on [PyTorch Homepage](https://pytorch.org/) for installing PyTorch (Python2). 
- The python packages required are ``` nltk ``` ``` tqdm ``` which can be installed using pip. </br>

### Data
To download the VQA dataset please use the script 'scripts/vqa_download.sh': </br>
```
scripts/vqa_download.sh `pwd`/data
```

### Process Data
Detailed instructions for processing data are provided by [GuessWhatGame/vqa](https://github.com/GuessWhatGame/vqa#introduction). </br>

#### Create dictionary
To create the VQA dictionary, use the script preprocess_data/create_dico.py. </br>
```
python preprocess_data/create_dictionary.py --data_dir data --year 2014 --dict_file dict.json
```

#### Create GLOVE dictionary
To create the GLOVE dictionary, download the original glove file and run the script preprocess_data/create_gloves.py. </br>
```
wget http://nlp.stanford.edu/data/glove.42B.300d.zip -P data/
unzip data/glove.42B.300d.zip -d data/
python preprocess_data/create_gloves.py --data_dir data --glove_in data/glove.42B.300d.txt --glove_out data/glove_dict.pkl --year 2014
```

### Train Model
To train the network, set the required parameters in ``` config.json ``` and run the script main.py.
```
python main.py --gpu gpu_id --data_dir data --img_dir images --config config.json --exp_dir exp --year 2014
```

### Citation
If you find this code useful, please consider citing the original work by authors:
```
@inproceedings{de2017modulating,
author = {Harm de Vries and Florian Strub and J\'er\'emie Mary and Hugo Larochelle and Olivier Pietquin and Aaron C. Courville},
title = {Modulating early visual processing by language},
booktitle = {Advances in Neural Information Processing Systems 30},
year = {2017}
url = {https://arxiv.org/abs/1707.00683}
}
```
