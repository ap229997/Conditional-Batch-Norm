import logging
import os

from data_provider.image_loader import get_img_builder
from data_provider.nlp_utils import GloveEmbeddings

from vqa_preprocess.vqa_tokenizer import VQATokenizer
from vqa_preprocess.vqa_dataset import VQADataset

import torch
from torch.autograd import Variable

import numpy as np

logger = logging.getLogger()

'''
DataLoader : to load the 
				- Train, Validation and Test data
				- Tokenizer
				- Glove Embeddings
'''
class DataLoader(object):

	def __init__(self, config, data_dir, img_dir, year, test_set, batch_size):
		super(DataLoader, self).__init__()

		self.batch_size = batch_size
		self.total_len = None # total size of the dataset being used
		self.use_glove = config['model']['glove']

		# Load images
		logger.info('Loading images..')
		self.image_builder = get_img_builder(config['model']['image'], img_dir)
		self.use_resnet = self.image_builder.is_raw_image()
		self.require_multiprocess = self.image_builder.require_multiprocess()

		# Load dictionary
		logger.info('Loading dictionary..')
		self.tokenizer = VQATokenizer(os.path.join(data_dir, config["dico_name"]))

		# Load data
		logger.info('Loading data..')
		self.trainset = VQADataset(data_dir, year=year, which_set="train", image_builder=self.image_builder, 
													preprocess_answers=self.tokenizer.preprocess_answers)
		self.validset = VQADataset(data_dir, year=year, which_set="val", image_builder=self.image_builder, 
													preprocess_answers=self.tokenizer.preprocess_answers)
		self.testset = VQADataset(data_dir, year=year, which_set=test_set, image_builder=self.image_builder)

		# Load glove
		self.glove = None
		if self.use_glove:
			logger.info('Loading glove..')
			self.glove = GloveEmbeddings(os.path.join(data_dir, config["glove_name"]))

	'''
	Arguments:
		ind : current iteration which is converted to required indices to be loaded
		data_type: specifies the train('train'), validation('val') and test('test') partition

	Returns:
		image : image in the form of torch.autograd.Varibale - (batch, channels, height, width) format
		tokens : question tokens
		glove_emb : glove embedding of the question
		answer : ground truth tokens
	'''
	def get_mini_batch(self, ind, data_type='train'):
		
		if data_type == 'train':
			dataset = self.trainset.games
		elif data_type == 'val':
			dataset = self.validset.games
		elif data_type == 'test':
			dataset = self.testset.games

		self.total_len = len(dataset) # total elements in dataset

		# specify the start and end indices of the minibatch
		# In case, the indices goes over total elements
		# wrap the indices around the dataset
		start_ind = (ind*self.batch_size)%self.total_len
		end_ind = ((ind+1)*self.batch_size)%self.total_len
		if start_ind < end_ind:
			data = dataset[start_ind:end_ind]
		else:
			data = dataset[start_ind:self.total_len]
			data.extend(dataset[0:end_ind])

		# get the images from the dataset and convert them into torch.autograd.Variable
		image = np.array([x.image.get_image() for x in data])
		image = torch.autograd.Variable(torch.Tensor(image).cuda())
		# reshape the image to (batch, channels, height, width) format
		image = image.permute(0,3,1,2).contiguous()

		# get the questions from the dataset, tokenize them and convert them into torch.autograd.Variable
		que = [x.question for x in data]
		tokens = [self.tokenizer.encode_question(x)[0] for x in que]
		words = [self.tokenizer.encode_question(x)[1] for x in que]
		max_len = max([len(x) for x in tokens]) # max length of the question
		# pad the additional length with unknown token '<unk>'
		for x in tokens:
			for i in range(max_len-len(x)):
				x.append(self.tokenizer.word2i['<unk>'])
		for x in words:
			for i in range(max_len-len(x)):
				x.append('<unk>')
		tokens = Variable(torch.LongTensor(tokens).cuda())
		
		# get the ground truth answer, tokenize them and convert them into torch.autograd.Variable
		ans = [x.majority_answer for x in data]
		answer = [self.tokenizer.encode_answer(x) for x in ans]
		answer = Variable(torch.LongTensor(answer).cuda())
		
		# get the glove embeddings of the question token and convert them into torch.autograd.Variable
		glove_emb = [self.glove.get_embeddings(x) for x in words]
		glove_emb = Variable(torch.Tensor(glove_emb).cuda())

		return image, tokens, glove_emb, answer