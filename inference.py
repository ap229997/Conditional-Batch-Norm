import logging
import os

from data_provider.image_loader import get_img_builder
from data_provider.nlp_utils import GloveEmbeddings

from vqa_preprocess.vqa_tokenizer import VQATokenizer
from vqa_preprocess.vqa_dataset import VQADataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

parser = argparse.ArgumentParser('VQA network inference!')

parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--img_path", type=str, help="path of image")
parser.add_argument("--config", type=str, help='Config file')
parser.add_argument("--load_checkpoint", type=str, help="Load model parameters from specified checkpoint")

args = parser.parse_args()

torch.cuda.set_device(args.gpu)

logger = logging.getLogger()

def load_config(config_file):
	with open(config_file, 'rb') as f_config:
		config_str = f_config.read()
		config = json.loads(config_str.decode('utf-8'))

	return config

config = load_config(args.config)
logger = logging.getLogger()

# required parameters
finetune = config["model"]["image"].get('finetune', list())
use_glove = config["model"]["glove"]
lstm_size = config["model"]["no_hidden_LSTM"]
emb_size = config["model"]["word_embedding_dim"]
resnet_model = config["model"]["image"]["resnet_version"]

'''
DataLoader : to load
				- single image for inference (can be modified for mini-batch)
				- Tokenizer
				- Glove Embeddings (if required)
'''
class DataLoader(object):

	def __init__(self, config, batch_size=1):
		super(DataLoader, self).__init__()

		self.batch_size = batch_size
		self.use_glove = config['model']['glove']

		# Load dictionary
		logger.info('Loading dictionary..')
		self.tokenizer = VQATokenizer(os.path.join(data_dir, config["dico_name"]))

		# Load glove
		self.glove = None
		if self.use_glove:
			logger.info('Loading glove..')
			self.glove = GloveEmbeddings(os.path.join(data_dir, config["glove_name"]))

	'''
	Arguments:
		image : pass the image in (batch, height, width, channels) format - batch=1 for single image
		question : list of question required

	Returns:
		image : image in the form of torch.autograd.Varibale - (batch, channels, height, width) format
		tokens : question tokens
		glove_emb : glove embedding of the question
		answer : ground truth tokens
	'''
	def process_data(self, image, question):

		# convert the image into torch.autograd.Variable
		image = torch.autograd.Variable(torch.Tensor(image).cuda())
		# reshape the image to (batch, channels, height, width) format
		image = image.permute(0,3,1,2).contiguous()

		# tokenize the questions and convert them into torch.autograd.Variable
		tokens = [self.tokenizer.encode_question(x)[0] for x in question]
		words = [self.tokenizer.encode_question(x)[1] for x in question]
		max_len = max([len(x) for x in tokens]) # max length of the question
		# pad the additional length with unknown token '<unk>'
		for x in tokens:
			for i in range(max_len-len(x)):
				x.append(self.tokenizer.word2i['<unk>'])
		for x in words:
			for i in range(max_len-len(x)):
				x.append('<unk>')
		tokens = Variable(torch.LongTensor(tokens).cuda())

		# if required
		# get the glove embeddings of the question token and convert them into torch.autograd.Variable
		glove_emb = [self.glove.get_embeddings(x) for x in words]
		glove_emb = Variable(torch.Tensor(glove_emb).cuda())

		return image, tokens, glove_emb

def main():
	dataloader = DataLoader(config, batch_size=1)

	model = Net(config=config, no_words=dataloader.tokenizer.no_words, no_answers=dataloader.tokenizer.no_answers,
				resnet_model=resnet_model, lstm_size=lstm_size, emb_size=emb_size, use_pretrained=False)

	model_weight = torch.load(config['checkpoint_path'])
	model.load_state_dict(model_weight, strict = False)
	model.cuda().eval()
	'''
	load the image and question in proper format and then execute the below code
	'''

	image, tokens, glove_emb = dataloader.process_data(image, question)

	ind_ans = model(image, tokens, glove_emb)

	# convert the ind_ans (indices to answers with max probability)
	# using decode_answer function in tokenizer (look it up once)

if __name__ == '__main__':
	main()
