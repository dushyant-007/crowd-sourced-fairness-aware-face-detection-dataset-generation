import os
import torch
import numpy as np
import glob
import matplotlib.pyplot as plt
import imageio as iio
from mlxtend.image import extract_face_landmarks
from PIL import Image

def convert(filename):
    img = Image.open(filename)
    pil_img = img.convert('RGB')
    arr_img = np.array(pil_img.getdata(), dtype=np.uint8).reshape(pil_img.height, pil_img.width, 3)
    l = []
    for row in arr_img:
        col = []
        for pixel in row:
            col.append(int(pixel[0]),int(pixel[1]),int(pixel[2]))
        l.append(col)
    return l

def save(img, filename):
    arr = np.asarray(img, dtype=np.uint8)
    pil_img = Image.fromarray(arr)
    pil_img.save(filename, format='png')


def cal_weights(csv, target):
	vals = torch.tensor([x[1] for x in sorted(csv[target].value_counts().items())])
	vals = 1 - (vals / vals.sum())

	return vals

# 'forehead', 'eyebrow', 'eye', 'mouth', 'chin', 'cheek', 'nose', 'ear', 'lip'(same with mouth); do 'skin' later 
def get_compo(image_path, compo='eye'):
	landmarks = extract_face_landmarks(iio.imread(image_path))
	x, y = []
	compo = compo.lower()

	if compo == 'forehead':
		x = [i for i in range(landmarks[17, 0], landmarks[26, 0])]
		y = [j for j in range(70, landmarks[19, 1] if landmarks[19, 1] > landmarks[24, 1] else landmarks[24, 1])]
	elif compo == 'eyebrow':
		x = [i for i in range(landmarks[17, 0], landmarks[21, 0])] + [x for x in range(landmarks[22, 0], landmarks[26, 0])]
		y = [j for j in range(landmarks[19, 1] if landmarks[19, 1] > landmarks[24, 1] else landmarks[24, 1], landmarks[17, 1] if landmarks[17, 1] < landmarks[22, 1] else landmarks[22, 1])]
	elif compo == 'eye':
		x = [i for i in range(landmarks[36, 0], landmarks[39, 0])] + [x for x in range(landmarks[42, 0], landmarks[45, 0])]
		y = [j for j in range(landmarks[38, 1] if landmarks[38, 1] < landmarks[44, 1] else landmarks[44, 1], landmarks[40, 1] if landmarks[40, 1] < landmarks[46, 1] else landmarks[46, 1])]
	elif compo == 'mouth' or compo == 'lip':
		x = [i for i in range(landmarks[48, 0], landmarks[54, 0])]
		y = [j for j in range(landmarks[49, 1] if landmarks[49, 1] > landmarks[53, 1] else landmarks[53, 1], landmarks[57, 1] if landmarks[57, 1] > landmarks[56, 1] else landmarks[56, 1])]
	elif compo == 'chin':
		x = [i for i in range(landmarks[5, 0], landmarks[11, 0])]
		y = [j for j in range(landmarks[57, 1], landmarks[8, 1])]
	elif compo == 'cheek':
		x = [i for i in range(landmarks[2, 0], landmarks[31, 0])] + [i for i in range(landmarks[35, 0], landmarks[14, 0])]
		y = [j for j in range(landmarks[41, 1], landmarks[6, 1])]
	elif compo == 'nose':
		x = [i for i in range(landmarks[31, 0], landmarks[35, 0])]
		y = [j for j in range(landmarks[33, 1], landmarks[27, 1])]
	elif compo == 'ear':
		x = [i for i in range(landmarks[14, 0], landmarks[14, 0]) + 10] + [i for i in range(landmarks[2, 0] - 10, landmarks[2, 0])]
		y = [j for j in range(landmarks[16, 1], landmarks[13, 1])]
	# else:		for skin


	img = Image.open(image_path)
	pil_img = img.convert('RGB')
	arr_img = np.array(pil_img.getdata(), dtype=np.uint8).reshape(pil_img.height, pil_img.width, 3)

	for i in range(len(arr_img)):
		for j in range(len(arr_img[i])):
			if not(j in x and i in y):
				arr_img[i][j] = np.array([0, 0, 0])
	pil_img = Image.fromarray(arr_img)
	pil_img.save(os.path.join(os.getcwd(), 'images\img.jpg'))
	# pil_img.save(os.path.join(os.getcwd(), image_path))

# get_compo('./dataset/celeba/images\1\000375.jpg')