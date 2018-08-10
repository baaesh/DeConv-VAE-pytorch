import torch
from torch import nn
import torch.nn.functional as F

from random import randint


def compute_cross_entropy(log_prob, target):
	# compute reconstruction loss using cross entropy
	loss = [F.nll_loss(sentence_emb_matrix, word_ids, size_average=False) for sentence_emb_matrix, word_ids in zip(log_prob, target)]
	average_loss = sum([torch.sum(l) for l in loss]) / log_prob.size()[0]
	return average_loss


def loss_function(log_prob, target, mu, logvar):
	reconst = compute_cross_entropy(log_prob, target)
	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

	return reconst, KLD


def test(model, data, mode='test'):
	with torch.no_grad():
		if mode == 'dev':
			iterator = iter(data.dev_iter)
		else:
			iterator = iter(data.test_iter)

		model.eval()
		test_reconst, test_KLD = 0, 0
		loss, size = 0, 0

		for batch in iterator:
			batch_text = torch.cat([batch.premise, batch.hypothesis], dim=0)
			log_prob, mu, logvar, _ = model(batch_text)

			reconst, KLD = loss_function(log_prob, batch_text, mu, logvar)
			batch_loss = reconst + KLD
			loss += batch_loss.item()

			test_reconst += reconst.item()
			test_KLD += KLD.item()
			size += 1

		test_reconst /= size
		test_KLD /= size
		loss /= size
		return loss, test_reconst, test_KLD


def snli_test(model, data, mode='test'):
	with torch.no_grad():
		if mode == 'dev':
			iterator = iter(data.dev_iter)
		else:
			iterator = iter(data.test_iter)

		criterion = nn.CrossEntropyLoss()
		model.eval()
		acc, loss, size = 0, 0, 0

		for batch in iterator:
			pred, _, _, _, _, _, _ = model(batch)

			batch_loss = criterion(pred, batch.label)
			loss += batch_loss.item()

			_, pred = pred.max(dim=1)
			acc += (pred == batch.label).sum().float()
			size += len(pred)

		acc /= size
		acc = acc.cpu().item()
		return loss, acc

# reconstuct an example from test set
def example(model, args, data):
	i = randint(0, len(data.test.examples))

	e = data.test.examples[i]

	print(e.premise)
	p = torch.ones(29, dtype=torch.long).to(torch.device(args.device))
	for i in range(len(e.premise)):
		if i < 29:
			p[i] = data.TEXT.vocab.stoi[e.premise[i]]

	model.eval()
	log_prob, mu, logvar, _ = model(p.unsqueeze(0))

	_, predict_index = torch.max(log_prob, 2)
	p_predict = [data.TEXT.vocab.itos[word] for word in predict_index[0]]

	print(p_predict)


def snli_example(model, args, data):
	i = randint(0, len(data.test.examples))

	e = data.test.examples[i]

	print(e.premise)
	print(e.hypothesis)
	p = torch.ones(29, dtype=torch.long).to(torch.device(args.device))
	h = torch.ones(29, dtype=torch.long).to(torch.device(args.device))
	for i in range(len(e.premise)):
		if i < 29:
			p[i] = data.TEXT.vocab.stoi[e.premise[i]]
	for i in range(len(e.hypothesis)):
		if i < 29:
			h[i] = data.TEXT.vocab.stoi[e.hypothesis[i]]

	example = object
	setattr(example, 'premise', p.unsqueeze(0))
	setattr(example, 'hypothesis', h.unsqueeze(0))

	model.eval()
	pred, p_log_prob, p_mu, p_logvar, z_p, h_log_prob, h_mu, h_logvar, z_h = model(example)

	_, p_predict_index = torch.max(p_log_prob, 2)
	_, h_predict_index = torch.max(h_log_prob, 2)
	p_predict = [data.TEXT.vocab.itos[word] for word in p_predict_index[0]]
	h_predict = [data.TEXT.vocab.itos[word] for word in h_predict_index[0]]

	print(p_predict)
	print(h_predict)


# generate 10 sentences
def generate(model, args, data, sample_num):
	log_prob = model.generate(sample_num)
	_, predict_index = torch.max(log_prob, 2)

	for sentence in predict_index:
		predict = [data.TEXT.vocab.itos[word] for word in sentence]
		print(predict)