import torch
import torch.nn as nn

'''
For variable length lstm use padding with unknown '<unk>' token
'''
class VariableLengthLSTM(nn.Module):

	def __init__(self, config):

		super(VariableLengthLSTM, self).__init__()

		self.num_hidden = int(config["no_hidden_LSTM"])
		self.depth = int(config["no_LSTM_cell"])
		self.word_emb_len = 2*int(config["word_embedding_dim"])

		self.lstm = nn.LSTM(input_size=self.word_emb_len, hidden_size=self.num_hidden, num_layers=self.depth, batch_first=True, dropout=0)

	def forward(self, word_emb):
		out = self.lstm(word_emb)

		return out