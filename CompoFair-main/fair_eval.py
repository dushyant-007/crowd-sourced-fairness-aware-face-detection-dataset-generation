import os
import numpy as np
import pandas as pd
from scipy.special import softmax


def eq_odds(gt, pred, attri):
	csv = pd.DataFrame({'gt': gt, 'pred': pred, 'attri': attri})
	res = 0
	for name, group in csv.groupby(['gt', 'attri']):
		score = len(group[group['pred'] == 1])/len(group)
		res += score if name[1] == 0 else -score

	return res


def eq_opp(gt, pred, attri):
	csv = pd.DataFrame({'gt': gt, 'pred': pred, 'attri': attri})
	csv = csv[csv['gt']==1]
	res = 0
	for name, group in csv.groupby(['gt', 'attri']):
		score = len(group[group['pred'] == 1])/len(group)
		res += score if name[1] == 0 else -score

	return res


def demo_parity(gt, pred, attri):
	csv = pd.DataFrame({'gt': gt, 'pred': pred, 'attri': attri})
	res = 0
	for name, group in csv.groupby(['attri']):
		score = len(group[group['pred']==1])/len(group)
		res += score if name == 0 else -score

	return res


def pred_parity(gt, pred, attri):
	csv = pd.DataFrame({'gt': gt, 'pred': pred, 'attri': attri})
	csv = csv[csv['pred']==1]
	res = 0
	for name, group in csv.groupby(['pred', 'attri']):
		score = len(group[group['gt'] == 1])/len(group)
		res += score if name[1] == 0 else -score

	return res


def calibration(gt, score, attri, dec=1):
	csv = pd.DataFrame({'gt': gt, 'score': np.around(score[:, 1], dec), 'attri': attri})
	res = 0
	for name, group in csv.groupby(['score', 'attri']):
		score = len(group[group['gt'] == 1])/len(group)
		res += score if name[1] == 0 else -score

	return res


def balance_score(gt, score, attri, plot=False):
	csv = pd.DataFrame({'gt': gt, 'score': score[:, 1], 'attri': attri})
	res = 0
	for name, group in csv.groupby(['gt', 'attri']):
		s = group['score'].mean()
		res += s if name[1] == 0 else -s

	if plot:
		print('developing...')
		raise

	return res


def fair_eval(score, gt, pred, attri):
	# print('eq odds: {}'.format(eq_odds(gt, pred, attri)))
	# print('eq opp: {}'.format(eq_opp(gt, pred, attri)))
	# print('demo parity: {}'.format(demo_parity(gt, pred, attri)))
	# print('pred parity: {}'.format(pred_parity(gt, pred, attri)))
	# print('calibration: {}'.format(calibration(gt, score, attri)))
	# print('balance for pos/neg class: {}'.format(balance_score(gt, score, attri)))

	return eq_odds(gt, pred, attri), eq_opp(gt, pred, attri), demo_parity(gt, pred, attri), pred_parity(gt, pred, attri), calibration(gt, score, attri), balance_score(gt, score, attri)


if __name__ == '__main__':
	score = softmax(np.random.normal(size=(1000, 2)), 1)
	pred = np.argmax(score, 1)
	gt = np.random.randint(0, 2, 1000)
	attri = np.random.randint(0, 2, 1000)
	fair_eval(score, pred, gt, attri)