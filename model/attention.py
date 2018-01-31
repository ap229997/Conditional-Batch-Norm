import torch
import torch.nn as nn

'''
Computes attention mechanism on the image feature map conditioned on the question embedding
'''
class Attention(nn.Module):

	def __init__(self, config):
		super(Attention, self).__init__()

		self.mlp_units = config['model']['image']['attention']['no_attention_mlp'] # hidden layer units of the MLP

		# MLP for concatenated feature map and question embedding
		self.fc = nn.Sequential(
            nn.Linear(3072, self.mlp_units),
            nn.ReLU(inplace=True),
            nn.Linear(self.mlp_units, 1),
            nn.ReLU(inplace=True),
            ).cuda()

		self.softmax = nn.Softmax2d() # to get the probablity values across the height and width of feature map

		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform(m.weight)
				nn.init.constant(m.bias, 0.1)

		self.batch_size = None
		self.channels = None
		self.height = None
		self.width = None
		self.len_emb = None

	'''
	Arguments: 
		feature : feature map from the block 4 of resnet
		lstm_emb : lstm embedding of the question

	Returns:
		soft_attention : soft attention on the feature map conditioned on the question embedding
	'''
	def forward(self, feature, lstm_emb):
		self.batch_size, self.channels, self.height, self.width = feature.data.shape
		self.len_emb = lstm_emb.data.shape[1]

		feature = feature.view(self.batch_size, self.channels, self.height*self.width)

		lstm_emb = torch.stack([lstm_emb]*self.height*self.width, dim=2)

		embedding = torch.cat([feature, lstm_emb], dim=1)
		embedding = embedding.permute(0,2,1).contiguous() # resize concatenated embedding according to MLP dimensions

		out = self.fc(embedding)
		out = out.permute(0,2,1).contiguous()
		out = out.view(self.batch_size, 1, self.height, self.width)
		# get the probability values across the height and width of the feature map
		alpha = self.softmax(out)

		feature = feature.view(self.batch_size, self.channels, self.height, self.width)
		soft_attention = feature * alpha
		soft_attention = soft_attention.view(self.batch_size, self.channels, -1)
		# mean pool across the height and width of the feature map with alpha value serving as weights
		soft_attention = torch.sum(soft_attention, dim=2)

		return soft_attention