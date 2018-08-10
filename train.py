import argparse
import copy
import os
import torch

from torch import nn, optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from time import gmtime, strftime

from model import NN4VAE, NN4SNLI
from data import SNLI
from test import test, snli_test, example, generate


def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_cross_entropy(log_prob, target):
	# compute reconstruction loss using cross entropy
	loss = [F.nll_loss(sentence_emb_matrix, word_ids, size_average=False) for sentence_emb_matrix, word_ids in zip(log_prob, target)]
	average_loss = sum([torch.sum(l) for l in loss]) / log_prob.size()[0]
	return average_loss


def loss_function(log_prob, target, mu, logvar):
	reconst = compute_cross_entropy(log_prob, target)
	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

	return reconst, KLD


def train(args, data):
	model = NN4VAE(args, data)
	model.to(torch.device(args.device))

	parameters = filter(lambda p: p.requires_grad, model.parameters())
	optimizer = optim.Adam(parameters, lr=args.learning_rate)
	print("number of all parameters: " + str(count_parameters(model)))

	writer = SummaryWriter(log_dir='runs/' + args.model_time)

	model.train()
	train_reconst, train_KLD = 0, 0
	loss, size, last_epoch = 0, 0, -1

	iterator = data.train_iter
	for i, batch in enumerate(iterator):
		present_epoch = int(iterator.epoch)
		if present_epoch == args.epoch:
			break
		if present_epoch > last_epoch:
			print('epoch:', present_epoch + 1)
			generate(model, args, data, sample_num=10)
		last_epoch = present_epoch

		batch_text = torch.cat([batch.premise, batch.hypothesis], dim=0)
		log_prob, mu, logvar, _ = model(batch_text)

		optimizer.zero_grad()
		reconst, KLD = loss_function(log_prob, batch_text, mu, logvar)
		batch_loss = reconst + KLD
		loss += batch_loss.item()
		batch_loss.backward()
		optimizer.step()

		train_reconst += reconst.item()
		train_KLD += KLD.item()
		size += 1

		writer.add_scalar('KL_divergence/train', KLD.item(), size)
		if (i + 1) % args.print_freq == 0:
			train_reconst /= size
			train_KLD /= size
			loss /= size

			dev_loss, dev_reconst, dev_KLD = test(model, data, mode='dev')
			test_loss, test_reconst, test_KLD = test(model, data)

			c = (i + 1) // args.print_freq

			writer.add_scalar('loss/train', loss, c)
			writer.add_scalar('reconstruction loss/train', train_reconst, c)

			writer.add_scalar('loss/dev', dev_loss, c)
			writer.add_scalar('reconstruction loss/dev', dev_reconst, c)
			writer.add_scalar('KL_divergence/dev', dev_KLD, c)
			writer.add_scalar('loss/test', test_loss, c)
			writer.add_scalar('reconstruction loss/test', test_reconst, c)
			writer.add_scalar('KL_divergence/test', test_KLD, c)

			print(f'train loss: {loss:.5f} / train reconstruction loss: {train_reconst:.5f} / train KL divergence: {train_KLD:.5f}')
			print(f'dev loss: {dev_loss:.5f} / dev reconstruction loss: {dev_reconst:.5f} / dev KL divergence: {dev_KLD:.5f}')
			print(f'test loss: {test_loss:.5f} / test reconstruction loss: {test_reconst:.5f} / test KL divergence: {test_KLD:.5f}')

			example(model, args, data)

			train_reconst, train_KLD, loss, size = 0, 0, 0, 0
			model.train()

	writer.close()

	return model


def snli_train(args, data):
	model = NN4SNLI(args, data)
	model.to(torch.device(args.device))

	parameters = filter(lambda p: p.requires_grad, model.parameters())
	optimizer = optim.Adam(parameters, lr=args.learning_rate)
	criterion = nn.CrossEntropyLoss()
	print("number of all parameters: " + str(count_parameters(model)))

	writer = SummaryWriter(log_dir='runs/' + args.model_time)

	model.train()
	acc, loss, size, last_epoch = 0, 0, 0, -1
	train_reconst, train_KLD, vae_size = 0, 0, 0
	max_dev_acc, max_test_acc = 0, 0
	alpha = torch.Tensor([-0.1]).to(torch.device(args.device))

	iterator = data.train_iter
	for i, batch in enumerate(iterator):
		present_epoch = int(iterator.epoch)
		if present_epoch == args.epoch:
			break
		if present_epoch > last_epoch:
			print('epoch:', present_epoch + 1)
			if alpha < 1:
				alpha += 0.1
		last_epoch = present_epoch

		pred, p_log_prob, p_mu, p_logvar, h_log_prob, h_mu, h_logvar = model(batch)

		optimizer.zero_grad()
		batch_loss = alpha * criterion(pred, batch.label)
		p_reconst, p_KLD = loss_function(p_log_prob, batch.premise, p_mu, p_logvar)
		h_reconst, h_KLD = loss_function(h_log_prob, batch.hypothesis, h_mu, h_logvar)
		batch_loss += p_reconst + h_reconst + p_KLD + h_KLD
		loss += batch_loss.item()
		batch_loss.backward()
		optimizer.step()

		train_reconst += p_reconst.item() + h_reconst.item()
		train_KLD += p_KLD.item() + h_KLD.item()
		vae_size += 2

		_, pred = pred.max(dim=1)
		acc += (pred == batch.label).sum().float()
		size += len(pred)

		if (i + 1) % args.print_freq == 0:
			acc /= size
			acc = acc.cpu().item()
			loss /= vae_size
			train_reconst /= vae_size
			train_KLD /= vae_size
			dev_loss, dev_acc = snli_test(model, data, mode='dev')
			test_loss, test_acc = snli_test(model, data)
			c = (i + 1) // args.print_freq

			writer.add_scalar('loss/train', loss, c)
			writer.add_scalar('acc/train', acc, c)
			writer.add_scalar('reconstruction loss/train', train_reconst, c)
			writer.add_scalar('KL_divergence/train', train_KLD, c)
			writer.add_scalar('loss/dev', dev_loss, c)
			writer.add_scalar('acc/dev', dev_acc, c)
			writer.add_scalar('loss/test', test_loss, c)
			writer.add_scalar('acc/test', test_acc, c)

			print(f'train loss: {loss:.5f} / train reconstruction loss: {train_reconst:.5f} / train KL divergence: {train_KLD:.5f}')
			print(f'dev loss: {dev_loss:.3f} / test loss: {test_loss:.3f}'
				  f' / train acc: {acc:.3f} / dev acc: {dev_acc:.3f} / test acc: {test_acc:.3f}')

			if dev_acc > max_dev_acc:
				max_dev_acc = dev_acc
				max_test_acc = test_acc
				best_model = copy.deepcopy(model)

			acc, loss, size = 0, 0, 0
			vae_size = 0
			model.train()

	writer.close()
	print(f'max dev acc: {max_dev_acc:.3f} / max test acc: {max_test_acc:.3f}')

	return best_model


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch-size', default=16, type=int)
	parser.add_argument('--data-type', default='SNLI')
	parser.add_argument('--dropout', default=0.3, type=float)
	parser.add_argument('--epoch', default=20, type=int)
	parser.add_argument('--gpu', default=0, type=int)
	parser.add_argument('--learning-rate', default=3e-4, type=float)
	parser.add_argument('--print-freq', default=3000, type=int)
	parser.add_argument('--word-dim', default=300, type=int)
	parser.add_argument('--filter-size', default=5, type=int)
	parser.add_argument('--stride', default=2, type=int)
	parser.add_argument('--latent-size', default=500, type=int)
	parser.add_argument('--tau', default=0.01, type=float)
	parser.add_argument('--hidden-size', default=500, type=int)
	parser.add_argument('--mode', default='VAE', help="available mode: VAE, SNLI")

	args = parser.parse_args()

	print('loading SNLI data...')
	data = SNLI(args)

	setattr(args, 'word_vocab_size', len(data.TEXT.vocab))
	setattr(args, 'class_size', len(data.LABEL.vocab))
	setattr(args, 'model_time', strftime('%H:%M:%S', gmtime()))
	setattr(args, 'feature_maps', [300, 600, 500])
	if args.gpu > -1:
		setattr(args, 'device', "cuda:0")
	else:
		setattr(args, 'device', "cpu")

	print('training start!')
	if args.mode == 'VAE':
		best_model = train(args, data)
	else:
		best_model = snli_train(args, data)

	if not os.path.exists('saved_models'):
		os.makedirs('saved_models')
	if args.mode == 'VAE':
		torch.save(best_model.state_dict(), f'saved_models/DeConv_VAE_{args.data_type}_{args.model_time}.pt')
	else:
		torch.save(best_model.state_dict(), f'saved_models/DeConv_VAE_SNLI_{args.data_type}_{args.model_time}.pt')

	print('training finished!')


if __name__ == '__main__':
	main()
