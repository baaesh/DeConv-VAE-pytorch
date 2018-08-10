import torch
import torch.nn as nn


class ConvolutionEncoder(nn.Module):

	def __init__(self, args):
		super(ConvolutionEncoder, self).__init__()

		self.conv1 = nn.Conv2d(1, args.feature_maps[0], (args.filter_size, args.word_dim), stride=args.stride)
		self.conv2 = nn.Conv2d(args.feature_maps[0], args.feature_maps[1], (args.filter_size, 1), stride=args.stride)
		self.conv3 = nn.Conv2d(args.feature_maps[1], args.feature_maps[2], (args.filter_size, 1), stride=args.stride)

		self.relu = nn.ReLU()


	def forward(self, x):
		# reshape for convolution layer
		x.unsqueeze_(1)

		h1 = self.relu(self.conv1(x))
		h2 = self.relu(self.conv2(h1))
		h3 = self.relu(self.conv3(h2))

		# (batch, feature_maps[2])
		h3.squeeze_()
		if len(h3.size()) < 2:
			h3.unsqueeze_(0)
		return h3


class DeconvolutionDecoder(nn.Module):

	def __init__(self, args):
		super(DeconvolutionDecoder, self).__init__()

		self.deconv1 = nn.ConvTranspose2d(args.latent_size, args.feature_maps[1], (args.filter_size, 1), stride=args.stride)
		self.deconv2 = nn.ConvTranspose2d(args.feature_maps[1], args.feature_maps[0], (args.filter_size, 1), stride=args.stride)
		self.deconv3 = nn.ConvTranspose2d(args.feature_maps[0], 1, (args.filter_size, args.word_dim), stride=args.stride)

		self.relu = nn.ReLU()


	def forward(self, z):
		# reshape for deconvolution layer
		z = z.unsqueeze(-1).unsqueeze(-1)

		h2 = self.relu(self.deconv1(z))
		h1 = self.relu(self.deconv2(h2))
		x_hat = self.relu(self.deconv3(h1))

		# (batch, seq_len, word_dim)
		x_hat.squeeze_()
		if len(x_hat.size()) < 3:
			x_hat.unsqueeze_(0)
		return x_hat


