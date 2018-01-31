import torch
import torch.nn as nn

from resnet import *
from var_len_lstm import VariableLengthLSTM
from attention import Attention

EPS = 1.0e-10

'''
creates the modified ResNet with CBN with the specified version

Arguments:
	n : resnet version [18, 34, 50, 101, 152]
	lstm_size : size of lstm embedding required for the CBN layer
	emb_size : size of hidden layer of MLP used to predict beta and gamma values

Returns:
	model : requried resnet model
'''
def create_resnet(n, lstm_size, emb_size, use_pretrained):
	if n == 18:
		model = resnet18(lstm_size, emb_size, pretrained=use_pretrained).cuda()
	if n == 34:
		model = resnet34(lstm_size, emb_size, pretrained=use_pretrained).cuda()
	if n == 50:
		model = resnet50(lstm_size, emb_size, pretrained=use_pretrained).cuda()
	if n == 101:
		model = resnet101(lstm_size, emb_size, pretrained=use_pretrained).cuda()
	if n == 152:
		model = resnet152(lstm_size, emb_size, pretrained=use_pretrained).cuda()

	return model

'''
VQA Architecture : 
			Consists of:
				- Embedding layer : to get the word embedding
				- Variable Length LSTM : used to get lstm representation of the question
										 embedding concatenated with the glove vectors
				- Resnet layer with CBN
				- Attention layer
				- MLP for question embedding
				- MLP for image embedding
				- Dropout
				- MLP for fused question and image embedding (element wise product)
				- Softmax Layer
				- Cross Entropy Loss
'''
class Net(nn.Module):

	def __init__(self, config, no_words, no_answers, resnet_model, lstm_size, emb_size, use_pretrained=True):
		super(Net, self).__init__()

		self.use_pretrained = use_pretrained # whether to use pretrained ResNet
		self.word_cnt = no_words # total count of words
		self.ans_cnt = no_answers # total count of valid answers
		self.lstm_size = lstm_size # lstm emb size to be passed to CBN layer
		self.emb_size = emb_size # hidden layer size of MLP used to predict delta beta and gamma parameters
		self.config = config # config file containing the values of parameters
		
		self.embedding = nn.Embedding(self.word_cnt, self.emb_size)
		self.lstm = VariableLengthLSTM(self.config['model']).cuda()
		self.net = create_resnet(resnet_model, self.lstm_size, self.emb_size, self.use_pretrained)
		self.attention = Attention(self.config).cuda()
		
		self.que_mlp = nn.Sequential(
						nn.Linear(config['model']['no_hidden_LSTM'], config['model']['no_question_mlp']),
						nn.Tanh(),
						)

		self.img_mlp = nn.Sequential(
						nn.Linear(2048, config['model']['no_image_mlp']),
						nn.Tanh(),
						)

		self.dropout = nn.Dropout(config['model']['dropout_keep_prob'])

		self.final_mlp = nn.Linear(config['model']['no_hidden_final_mlp'], self.ans_cnt)

		self.softmax = nn.Softmax()

		self.loss = nn.CrossEntropyLoss()

	'''
	Computes a forward pass through the network

	Arguments:
		image : input image
		tokens : question tokens
		glove_emb : glove embedding of the question
		labels : ground truth tokens

	Retuns: 
		loss : hard cross entropy loss
	'''
	def forward(self, image, tokens, glove_emb, labels):

		####### Question Embedding #######
		# get the lstm representation of the final state at time t
		que_emb = self.embedding(tokens)
		emb = torch.cat([que_emb, glove_emb], dim=2)
		lstm_emb, internal_state = self.lstm(emb)
		lstm_emb = lstm_emb[:,-1,:]

		####### Image features using CBN ResNet with Attention ########
		feature = self.net(image, lstm_emb)
		# l2 normalisation
		sq_sum = torch.sqrt(torch.sum(feature**2, dim=1)+EPS)
		sq_sum = torch.stack([sq_sum]*feature.data.shape[1], dim=1)
		feature = feature / sq_sum
		attn_feature = self.attention(feature, lstm_emb)

		####### MLP for question and image embedding ########
		lstm_emb = lstm_emb.view(feature.data.shape[0], -1)
		que_embedding = self.que_mlp(lstm_emb)
		image_embedding = self.img_mlp(attn_feature) 

		####### MLP for fused question and image embedding ########
		full_embedding = que_embedding * image_embedding
		full_embedding = self.dropout(full_embedding)
		out = self.final_mlp(full_embedding)
		
		prob = self.softmax(out)
		val, ind = torch.max(prob, dim=1)
		# hard cross entropy loss
		loss = self.loss(prob, labels)

		return loss

'''
# testing code
if __name__ == '__main__':
	torch.cuda.set_device(int(sys.argv[1]))
	net = Net(18, 512, 256)
'''