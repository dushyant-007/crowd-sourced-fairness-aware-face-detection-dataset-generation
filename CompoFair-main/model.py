import os
import io
import pdb
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import det_curve
from sklearn.metrics import recall_score

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1

import pytorch_lightning as pl
      

class FRBaseline(pl.LightningModule):
	def __init__(self, opt):
		super(FRBaseline, self).__init__()
		self.opt = opt
		self.wandb_logger = None
		self.num_classes = 3984
		self.backbone, out_feat = self.get_backbone()
		self.classifier = nn.Sequential(
			nn.Linear(out_feat, self.num_classes)
		)
		self.loss_func = nn.CrossEntropyLoss()
		self.cos_func = nn.CosineEmbeddingLoss()

	def forward(self, image):
		### IMAGE #####
		image = self.backbone(image)
		image = F.normalize(image, p=2, dim=1)
		image = self.classifier(image)

		return image

	def get_backbone(self):
		if self.opt.backbone == 'inceptionresnetv1':
			backbone = InceptionResnetV1(pretrained='vggface2') if self.opt.pretrain else InceptionResnetV1()
			out_feat = 512
		else:
			raise

		return backbone, out_feat


	def training_step(self, batch, batch_idx):
		imgs, labels, paths = batch
		preds = self.forward(imgs)
		loss = self.loss_func(preds, labels)

		self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(preds))
		return loss

	def validation_step(self, batch, batch_idx):
		imgs1, imgs2, labels, _, _ = batch
		preds1, preds2 = self.backbone(imgs1), self.backbone(imgs2)

		self.log("val_loss", self.cos_func(preds1, preds2, target=labels), on_step=True, on_epoch=True, prog_bar=True, logger=True)

	def test_step(self, batch, batch_idx):
		imgs1, imgs2, labels, male1, male2 = batch
		preds1, preds2 = self.backbone(imgs1), self.backbone(imgs2)

		return preds1, preds2, labels, male1, male2

	def test_epoch_end(self, outputs):
		preds1 = torch.cat([x[0] for x in outputs], 0)
		preds2 = torch.cat([x[1] for x in outputs], 0)
		labels = torch.cat([x[2] for x in outputs], 0)
		male1 = torch.cat([x[3] for x in outputs], 0)
		male2 = torch.cat([x[4] for x in outputs], 0)

		plt.bar(x=['overall_loss', 'male_loss', 'female_loss'], height=[
			self.cos_func(preds1, preds2, target=labels).cpu().numpy(),
			self.cos_func(preds1[(male1==1)&(male2==1)], preds2[(male1==1)&(male2==1)], target=labels[(male1==1)&(male2==1)]).cpu().numpy(),
			self.cos_func(preds1[(male1==-1)&(male2==-1)], preds2[(male1==-1)&(male2==-1)], target=labels[(male1==-1)&(male2==-1)]).cpu().numpy()
		])
		self.wandb_logger.experiment.log({"test/cos loss": plt})
		plt.close()

		# self.wandb_logger.experiment.log({'test/overall_loss': self.cos_func(preds1, preds2, target=labels)}, step=0)
		# self.wandb_logger.experiment.log({'test/male_loss': self.cos_func(preds1[(male1==1)&(male2==1)], preds2[(male1==1)&(male2==1)], target=labels[(male1==1)&(male2==1)])}, step=0)
		# self.wandb_logger.experiment.log({'test/female_loss': self.cos_func(preds1[(male1==-1)&(male2==-1)], preds2[(male1==-1)&(male2==-1)], target=labels[(male1==-1)&(male2==-1)])}, step=0)

		cos_sims = F.cosine_similarity(preds1, preds2, 1)

		cos_sims = cos_sims.cpu().numpy()
		labels = labels.cpu().numpy()
		male1 = male1.cpu().numpy()
		male2 = male2.cpu().numpy()

		thds = np.arange(0, 1, 0.1)
		fpr, fnr = self.get_fpr_fnr(cos_sims, labels, thds)
		male_fpr, male_fnr = self.get_fpr_fnr(cos_sims[(male1==1)&(male2==1)], labels[(male1==1)&(male2==1)], thds)
		female_fpr, female_fnr = self.get_fpr_fnr(cos_sims[(male1==-1)&(male2==-1)], labels[(male1==-1)&(male2==-1)], thds)

		plt.plot(thds, fpr, label='overall fpr')
		plt.plot(thds, male_fpr, label='male fpr')
		plt.plot(thds, female_fpr, label='female fpr')
		self.wandb_logger.experiment.log({"test/fpr": plt})
		plt.close()

		plt.plot(thds, fnr, label='overall fnr')
		plt.plot(thds, male_fnr, label='male fnr')
		plt.plot(thds, female_fnr, label='female fnr')
		self.wandb_logger.experiment.log({"test/fnr": plt})
		plt.close()

		thds = np.arange(0, 1, 0.1)
		acc = self.get_acc(cos_sims, labels, thds)
		male_acc = self.get_acc(cos_sims[(male1==1)&(male2==1)], labels[(male1==1)&(male2==1)], thds)
		female_acc = self.get_acc(cos_sims[(male1==-1)&(male2==-1)], labels[(male1==-1)&(male2==-1)], thds)

		plt.plot(thds, acc, label='overall acc')
		plt.plot(thds, male_acc, label='male acc')
		plt.plot(thds, female_acc, label='female acc')
		self.wandb_logger.experiment.log({"test/acc": plt})
		plt.close()

	def get_fpr_fnr(self, preds, labels, thds):
		fprs, fnrs = [], []
		for thd in thds:
			_preds = [1 if x >= thd else -1 for x in preds]
			tpr = recall_score(labels, _preds)
			tnr = recall_score(labels, _preds, pos_label = -1) 
			fpr = 1 - tnr
			fnr = 1 - tpr
			fprs.append(fpr)
			fnrs.append(fnr)

		return fprs, fnrs


	def get_acc(self, preds, labels, thds):
		accs = []
		for thd in thds:
			_preds = np.array([1 if x >= thd else -1 for x in preds])
			accs.append(np.mean((_preds==labels)))

		return accs


	def configure_optimizers(self):
		params = [
                {'params': self.classifier.parameters()},
                {'params': self.backbone.parameters(), 'lr': self.opt.base_lr/10}
        	]

		return torch.optim.Adam(params, lr=self.opt.base_lr)


class SegBaseline(pl.LightningModule):
	def __init__(self, opt):
		super(SegBaseline, self).__init__()
		self.opt = opt
		self.wandb_logger = None
		self.backbone, out_feat = self.get_backbone()
		self.classifier = nn.Sequential(
			nn.Linear(out_feat, 2)
		)
		self.loss_func = nn.BCEWithLogitsLoss()

	def forward(self, segs1, segs2):
		### IMAGE #####
		outs1, outs2 = self.backbone(segs1), self.backbone(segs2)
		outs = self.classifier(torch.cat([outs1, outs2], 1))

		return outs

	def get_backbone(self):
		if self.opt.backbone == 'inceptionresnetv1':
			backbone = InceptionResnetV1(pretrained='vggface2') if self.opt.pretrain else InceptionResnetV1()
			out_feat = 1024
		else:
			raise

		return backbone, out_feat


	def training_step(self, batch, batch_idx):
		segs1, segs2, labels = batch
		outs = self.forward(segs1, segs2)

		loss = self.loss_func(outs, labels)
		self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(outs))
		return loss

	def validation_step(self, batch, batch_idx):
		segs1, segs2, labels = batch
		outs = self.forward(segs1, segs2)

		self.log("val_loss", self.loss_func(outs, labels), on_step=True, on_epoch=True, prog_bar=True, logger=True)

	def test_step(self, batch, batch_idx):
		segs1, segs2, labels = batch
		outs = self.forward(segs1, segs2)

		return outs, labels

	def test_epoch_end(self, outputs):
		preds = torch.cat([x[0] for x in outputs], 0).cpu().numpy()
		labels = torch.cat([x[1] for x in outputs], 0).cpu().numpy()

		identity_preds, gender_preds = preds[:, 0], preds[:, 1]
		identity_labels, gender_labels = labels[:, 0], labels[:, 1]

		thds = list(np.arange(identity_preds.min(), identity_preds.max(), (identity_preds.max()-identity_preds.min())/10)) + [identity_preds.max()]
		identity_fpr, identity_fnr = self.get_fpr_fnr(identity_preds, identity_labels, thds)
		plt.plot(thds, identity_fpr, label='identity fpr')
		plt.plot(thds, identity_fnr, label='identity fnr')
		self.wandb_logger.experiment.log({"test/identity": plt})
		plt.close()

		identity_acc = self.get_acc(identity_preds, identity_labels, thds)
		plt.plot(thds, identity_acc, label='identity acc')
		self.wandb_logger.experiment.log({"test/identity_acc": plt})
		plt.close()

		thds = list(np.arange(gender_preds.min(), gender_preds.max(), (gender_preds.max()-gender_preds.min())/10)) + [gender_preds.max()]
		gender_fpr, gender_fnr = self.get_fpr_fnr(gender_preds, gender_labels, thds)
		plt.plot(thds, gender_fpr, label='gender fpr')
		plt.plot(thds, gender_fnr, label='gender fnr')
		self.wandb_logger.experiment.log({"test/gender": plt})
		plt.close()

		gender_acc = self.get_acc(gender_preds, gender_labels, thds)
		plt.plot(thds, gender_acc, label='gender acc')
		self.wandb_logger.experiment.log({"test/gender_acc": plt})
		plt.close()

	def get_fpr_fnr(self, preds, labels, thds):
		fprs, fnrs = [], []
		for thd in thds:
			_preds = [1 if x >= thd else 0 for x in preds]
			tpr = recall_score(labels, _preds)
			tnr = recall_score(labels, _preds, pos_label = 0) 
			fpr = 1 - tnr
			fnr = 1 - tpr
			fprs.append(fpr)
			fnrs.append(fnr)

		return fprs, fnrs


	def get_acc(self, preds, labels, thds):
		accs = []
		for thd in thds:
			_preds = np.array([1 if x >= thd else 0 for x in preds])
			accs.append(np.mean((_preds==labels)))

		return accs

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.opt.base_lr)