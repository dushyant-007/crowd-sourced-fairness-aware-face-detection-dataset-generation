from dataclasses import replace
import os
import re
import glob
from black import diff
import numpy as np
from tqdm import tqdm
import pandas as pd
import shutil

def gen_celeba_csv(images='./dataset/celeba/images', identity='./dataset/celeba/identity_CelebA.txt'):
	imgs = [x for x in glob.glob(os.path.join(images, '*'))]
	csv = pd.DataFrame(imgs, columns=['path'])
	csv['id'] = csv['path'].str.split('/').str[-1]
	identity = {line.split()[0]:line.split()[1].strip() for line in open(identity).readlines()}
	attributes = [line.strip().split() for line in open('./dataset/celeba/list_attr_celeba.txt').readlines()[2:]]
	attri_cols = open('./dataset/celeba/list_attr_celeba.txt').readlines()[1].strip().split(' ')
	csv['identity'] = csv['id'].map(identity)
	assert not csv['identity'].isnull().any()
	attributes = pd.DataFrame(attributes, columns=['id']+attri_cols)
	csv = csv.merge(attributes, on='id')
	csv[csv=='-1'] = 0
	csv.to_csv('./dataset/celeba/data.csv', index=False)


def gen_celeba_embeds(csv='./dataset/celeba/data.csv'):
	import face_recognition
	csv = pd.read_csv(csv)
	for path in tqdm(csv['path']):
		img = face_recognition.load_image_file(path)
		try:
			embed = face_recognition.face_encodings(img)[0]
		except:
			continue
		save = path.replace('images', 'embeds')+'.npy'
		folder = '/'.join(save.split('/')[:-1])
		if not os.path.exists(folder):
			os.makedirs(folder)

		np.save(save, embed)


def build_names():
	# creates a dictionary for each image (key) and the person appears on it (value)
	names_path = './dataset/celeba/identity_CelebA.txt'
	names = {}
	with open(os.path.join(os.getcwd(), names_path), 'r') as f:
		data = f.readlines()
		for i in data:
			pair = i.split()
			names.update({pair[0] : pair[1]})
	return names

# create a global variable that contains all people's names
all_names = build_names()


def gen_img_batch(B = 3, names = all_names):
	import random

	# all image feat Nxd and the path for each image
	images ='./dataset/celeba/images'
	paths = [x for x in glob.glob(os.path.join(images, '*', '*')) if len(x.split('/')[-1].split('\\')[-1]) == 10]

	while True:
		# randomly select B images from paths
		l = random.sample(paths, B)

		duplicate_names = {}
		for i in l:
			img = i.split('/')[-1].split('\\')[-1]
			each_name = names.get(img)
			if each_name in duplicate_names:
				continue
			else:
				duplicate_names.update({each_name : 1})
		break
	return l



	# create empty list L=[]
	# L = []


	# # all image feat Nxd and the path for each image
	# embeddings ='dataset/celeba/embeds'
	# paths = [i for i in glob.glob(os.path.join(embeddings, '*', '*'))]
	# imgs = [np.load(i) for i in glob.glob(os.path.join(embeddings, '*', '*'))]

	# # randomly select one image feat x=1xd, and add x to L
	# x = random.choice(imgs)
	# L.append(x)

	# # randomly sample value v as similarity from a gaussian
	# # and convert it into a percentage value (out of 6)
	# Norm_val = np.random.normal(0, 1, 1)
	# if Norm_val[0] > 3:
	# 	Norm_val[0] = 3
	# elif Norm_val[0] < -3:
	# 	Norm_val[0] = -3
	# similarity = (Norm_val[0] + 3) / 6

	# # calculate cosine similarity for each imgage in imgs
	# img_close_to_x = [1 - spatial.distance.cosine(x, i) for i in imgs]

	# # select the image feat y from Nxd that has the closest similarity with the x
	# #		1) y does not has the same identity as x
	# #		2) y is not in the batch
	# # add y to L
	# # until the number of images in L reaches B
	# for i in range(1, B):
	# 	min_similarity = 1
	# 	idx = 0
	# 	for index, value in enumerate(img_close_to_x):
	# 		difference = np.abs(value - similarity) 
	# 		if difference < min_similarity:
	# 			duplicate = False
	# 			for arr in L:
	# 				# find the image path for both arr in L 
	# 				# and the current image from img_close_to_x
	# 				arr_index = find_index(arr, imgs)
	# 				arr_file = paths[arr_index].split('/')[-1].split('\\')[-1][:-4]
	# 				img_file = paths[index].split('/')[-1].split('\\')[-1][:-4]
	# 				# if either the two images are identical or they refer to the same person,
	# 				# skip the current image from img_close_to_x
	# 				if np.array_equal(arr, imgs[index]) or names.get(arr_file) == names.get(img_file):
	# 					duplicate = True
	# 					break
	# 			if duplicate:
	# 				continue
	# 			min_similarity = difference
	# 			idx = index
	# 	L.append(imgs[idx])

	# # trace relative image path of each image feat in L and
	# # return paths for the selected images
	# return [path.replace('embeds', 'images')[:-4] for i in L for path in paths if np.array_equal(np.load(path), i)]		


# def find_index(ndarray, l):
# 	for idx, arr in enumerate(l):
# 		if np.array_equal(arr, ndarray):
# 			return idx


def gen_batches(num=1000):
	if len(os.listdir('./dataset/celeba/pool')) == 0:
		res = []
		for _ in range(num):
			_res = []
			imgs = gen_img_batch(B=3)
			for x in imgs:
				_x = '/'.join(x.split('/')[:-2] + x.split('/')[-1:])
				shutil.copy(x, _x.replace('images', 'pool'))
				_res.append(_x.replace('images', 'pool'))
			res.append(_res)
		res = pd.DataFrame(res, columns=['img1', 'img2', 'img3'])
		res.to_csv('./dataset/celeba/pool.csv', index=False)
	else:
		csv = pd.read_csv('./dataset/celeba/pool.csv')
		csv.sample(30).to_csv('./dataset/celeba/_pool.csv', index=False)


def gen_face_data_csv():
	def get_valid_adjs(csv):
		maps = csv['adj1'].value_counts().to_dict()
		maps.update(csv['adj2'].value_counts().to_dict())
		adjs = [k for k, v in maps.items() if v > 10]

		return adjs

	idx2compo = {i+1:x for i, x in enumerate(['forehead', 'eyebrow', 'eye', 'mouth', 'chin', 'cheek', 'nose', 'ear', 'temple', 'nostril', 'tooth', 'lip', 'tongue', 'skin'])}

	csv = pd.read_csv('./BatchData/cleaned_batch.csv')
	csv = csv[csv['Reject'].isnull()]
	csv = csv[['Input.img1', 'Input.img2', 'Input.img3', 'Answer.a1', 'Answer.a2', 'Answer.compo', 'Answer.d1', 'Answer.d2']].rename(
		columns={'Input.img1': 'img1', 'Input.img2': 'img2', 'Input.img3': 'img3',
		'Answer.a1': 'a1', 'Answer.a2': 'a2', 'Answer.compo': 'compo', 'Answer.d1': 'adj1', 'Answer.d2': 'adj2'}
	)
	csv['compo'] = csv['compo'].map(idx2compo)
	for adj in ['adj1', 'adj2']:
		csv[adj] = csv[adj].str.lower()
		csv[adj] = csv[adj].apply(lambda x: re.sub(r'[^\w\s]', '', x))
		csv[adj] = csv[adj].str.replace('white', 'light')
		csv[adj] = csv[adj].str.replace('lighter', 'light')
		csv[adj] = csv[adj].str.replace('darker', 'dark')
		csv[adj] = csv[adj].str.replace('black', 'dark')

	# with open('./dataset/celeba/valid_adjs.txt', 'w') as f:
	# 	for x in set(csv['adj1'].tolist()+csv['adj2'].tolist()):
	# 		f.write('{}\n'.format(x))

	valid_adjs = get_valid_adjs(csv)
	csv = csv[csv['adj1'].isin(valid_adjs)]
	csv = csv[csv['adj2'].isin(valid_adjs)]
	csv['a1'] -= 1
	csv['a2'] -= 1
	csv = csv[csv['a1']!=csv['a2']]

	csv = csv.groupby('compo').filter(lambda x: len(x)>10)
	for name, group in csv.groupby('compo'):
		print('------{}'.format(name))
		adjs = set(tuple(zip(group['adj1'], group['adj2'])))
		print('{} {} {} adjs for description'.format(name, len(group), len(adjs)))
		print(adjs)
	csv.to_csv('./dataset/celeba/crowd.csv', index=False)

if __name__ == '__main__':
	gen_celeba_csv()
	# gen_celeba_embeds()
	# gen_batches()
	# gen_face_data_csv()