import argparse
import logging
import os
import json

from model.net import Net
from model.dataloader import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import time

parser = argparse.ArgumentParser('VQA network baseline!')

parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--data_dir", type=str, help="Directory with data")
parser.add_argument("--img_dir", type=str, help="Directory with image")
parser.add_argument("--year", type=str, help="VQA release year (either 2014 or 2017)")
parser.add_argument("--test_set", type=str, default="test-dev", help="VQA release year (either 2014 or 2017)")
parser.add_argument("--exp_dir", type=str, help="Directory in which experiments are stored")
parser.add_argument("--config", type=str, help='Config file')
parser.add_argument("--load_checkpoint", type=str, help="Load model parameters from specified checkpoint")
parser.add_argument("--gpu_ratio", type=float, default=0.95, help="How many GPU ram is required? (ratio)")

args = parser.parse_args()

torch.cuda.set_device(args.gpu)

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
max_iters = config["optimizer"]["max_iters"]
lstm_size = config["model"]["no_hidden_LSTM"]
emb_size = config["model"]["word_embedding_dim"]
resnet_model = config["model"]["image"]["resnet_version"]
lr = config['optimizer']['learning_rate']
batch_size = config['optimizer']['batch_size']
step_size = config['optimizer']['step_size']

'''
Arguments:
	dataloader : to load images, tokens abs glove embeddings
	model : main VQA network
	optimizer : Adam (preferred)

Returns:
	None
'''
def train(dataloader, model, optimizer):

	model.train()
	iteration = 0

	while iteration < max_iters:
		st = time.time()
		image, tokens, glove_emb, answer = dataloader.get_mini_batch(iteration, data_type='train')

		optimizer.zero_grad()
		loss = model(image, tokens, glove_emb, answer)
		loss.backward()
		optimizer.step()

		iteration += 1

		print 'iteration: ', iteration, 'loss: ', loss.data[0], 'time taken: ', time.time()-st

		# save model state
		if iteration % 2000 == 0:
			torch.save(model.state_dict(), os.path.join(args.exp_dir, 'iter_%s.pth'%str(iteration)))

		# decrease learning rate by 10 after each step size
		if iteration % step_size == 0:
			lr = lr*0.1
			for param_group in optimizer.param_groups:
				param_group['lr'] = lr


def main():
	dataloader = DataLoader(config, args.data_dir, args.img_dir, args.year, args.test_set, batch_size)

	model = Net(config=config, no_words=dataloader.tokenizer.no_words, no_answers=dataloader.tokenizer.no_answers,
				resnet_model=resnet_model, lstm_size=lstm_size, emb_size=emb_size, use_pretrained=False).cuda()
	
	optimizer = optim.Adam(model.parameters(), lr=lr)

	train(dataloader, model, optimizer)

if __name__ == '__main__':
	main()