import torch
import torch.nn as nn
import torch.nn.functional as F

from module import *

class DeConvVAE(nn.Module):

	def __init__(self, args, data):
		super(DeConvVAE, self).__init__()
		self.args = args

		self.word_emb = nn.Embedding(args.word_vocab_size, args.word_dim)
		# initialize word embedding with GloVe
		self.word_emb.weight.data.copy_(data.TEXT.vocab.vectors)
		# fine-tune the word embedding
		self.word_emb.weight.requires_grad = True
		# <unk> vectors is randomly initialized
		nn.init.uniform_(self.word_emb.weight.data[0], -0.05, 0.05)

		self.encoder = ConvolutionEncoder(args)

		self.fc_mu = nn.Linear(args.feature_maps[2], args.latent_size)
		self.fc_logvar	= nn.Linear(args.feature_maps[2], args.latent_size)

		self.decoder = DeconvolutionDecoder(args)

		self.dropout = nn.Dropout(args.dropout)


	def reparameterize(self, mu, logvar):
		if self.training:
			std = torch.exp(0.5 * logvar)
			eps = torch.randn_like(std)
			return eps.mul(std).add_(mu)
		else:
			return mu


	def forward(self, x):
		# word embedding
		x = self.word_emb(x)

		# Encode
		h = self.encoder(self.dropout(x))
		mu = self.fc_mu(self.dropout(h))
		logvar = self.fc_logvar(self.dropout(h))

		# Sample
		z = self.reparameterize(mu, logvar)

		# Decode
		x_hat = self.decoder(z)

		# normalize
		norm_x_hat = torch.norm(x_hat, 2, dim=2, keepdim=True)
		rec_x_hat = x_hat / norm_x_hat
		norm_w = torch.norm(self.word_emb.weight.data, 2, dim=1, keepdim=True)
		rec_w = (self.word_emb.weight.data / (norm_w + 1e-20)).t()

		# compute probability
		prob_logits = torch.bmm(rec_x_hat, rec_w.unsqueeze(0)
								.expand(rec_x_hat.size(0), *rec_w.size())) / self.args.tau
		log_prob = F.log_softmax(prob_logits, dim=2)

		return log_prob, mu, logvar


	def generate(self, sample_num):
		latent_size = self.args.latent_size
		device = torch.device(self.args.device)

		# Sample
		z = torch.cat([torch.randn(latent_size).unsqueeze_(0) for i in range(sample_num)], dim=0)
		z = z.to(device)

		# Decode
		x_hat  = self.decoder(z)

		# normalize
		norm_x_hat = torch.norm(x_hat, 2, dim=2, keepdim=True)
		rec_x_hat = x_hat / norm_x_hat
		norm_w = torch.norm(self.word_emb.weight.data, 2, dim=1, keepdim=True)
		rec_w = (self.word_emb.weight.data / (norm_w + 1e-20)).t()

		# compute probability
		prob_logits = torch.bmm(rec_x_hat, rec_w.unsqueeze(0)
								.expand(rec_x_hat.size(0), *rec_w.size())) / self.args.tau
		log_prob = F.log_softmax(prob_logits, dim=2)

		return log_prob