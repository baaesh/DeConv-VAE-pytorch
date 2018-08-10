import torch
import torch.nn as nn
import torch.nn.functional as F

from module import *


class DeConvVAE(nn.Module):

	def __init__(self, args, data):
		super(DeConvVAE, self).__init__()
		self.args = args

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


	def forward(self, x, word_emb):
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
		norm_w = torch.norm(word_emb.weight.data, 2, dim=1, keepdim=True)
		rec_w = (word_emb.weight.data / (norm_w + 1e-20)).t()

		# compute probability
		prob_logits = torch.bmm(rec_x_hat, rec_w.unsqueeze(0)
								.expand(rec_x_hat.size(0), *rec_w.size())) / self.args.tau
		log_prob = F.log_softmax(prob_logits, dim=2)

		return log_prob, mu, logvar, z


	def generate(self, sample_num, word_emb):
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
		norm_w = torch.norm(word_emb.weight.data, 2, dim=1, keepdim=True)
		rec_w = (word_emb.weight.data / (norm_w + 1e-20)).t()

		# compute probability
		prob_logits = torch.bmm(rec_x_hat, rec_w.unsqueeze(0)
								.expand(rec_x_hat.size(0), *rec_w.size())) / self.args.tau
		log_prob = F.log_softmax(prob_logits, dim=2)

		return log_prob


class NN4VAE(nn.Module):

	def __init__(self, args, data):
		super(NN4VAE, self).__init__()

		self.args = args

		self.word_emb = nn.Embedding(args.word_vocab_size, args.word_dim)
		# initialize word embedding with GloVe
		self.word_emb.weight.data.copy_(data.TEXT.vocab.vectors)
		# fine-tune the word embedding
		self.word_emb.weight.requires_grad = True
		# <unk> vectors is randomly initialized
		nn.init.uniform_(self.word_emb.weight.data[0], -0.05, 0.05)

		self.vae = DeConvVAE(args, data)


	def forward(self, x):
		# word embedding
		x = self.word_emb(x)

		log_prob, mu, logvar, z = self.vae(x, self.word_emb)

		return log_prob, mu, logvar, z


	def generate(self, sample_num):
		return self.vae.generate(sample_num, self.word_emb)


class NN4SNLI(nn.Module):

	def __init__(self, args, data):
		super(NN4SNLI, self).__init__()

		self.args = args

		self.word_emb = nn.Embedding(args.word_vocab_size, args.word_dim)
		# initialize word embedding with GloVe
		self.word_emb.weight.data.copy_(data.TEXT.vocab.vectors)
		# fine-tune the word embedding
		self.word_emb.weight.requires_grad = True
		# <unk> vectors is randomly initialized
		nn.init.uniform_(self.word_emb.weight.data[0], -0.05, 0.05)

		self.vae = DeConvVAE(args, data)

		self.fc_1 = nn.Linear(4*args.latent_size, args.hidden_size)
		self.fc_2 = nn.Linear(args.hidden_size, args.hidden_size)
		self.fc_out = nn.Linear(args.hidden_size, args.class_size)

		self.relu = nn.ReLU()


	def forward(self, batch):
		p = batch.premise
		h = batch.hypothesis

		# (batch, seq_len, word_dim)
		p_x = self.word_emb(p)
		h_x = self.word_emb(h)

		# VAE
		p_log_prob, p_mu, p_logvar, z_p = self.vae(p_x, self.word_emb)
		h_log_prob, h_mu, h_logvar, z_h = self.vae(h_x, self.word_emb)

		# matching layer
		m = torch.cat([z_p, z_h, z_p - z_h, z_p * z_h], dim=-1)

		# fully-connected layers
		out = self.relu(self.fc_1(m))
		out = self.relu(self.fc_2(out))
		out = self.fc_out(out)

		return out, p_log_prob, p_mu, p_logvar, h_log_prob, h_mu, h_logvar