import torch, torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

from TemperaturePredNet import Net
from SuperUtils import DataUtils, TrainUtils

import matplotlib.pyplot as plt
from PIL import Image

config = {
	'loss_fn': 'MSELoss',
	'epochs': 10,
	'img_shape': (3, 510, 511),
	'device': 'cuda',
	'batch_size': 16,
	'lr': 0.001
		}

data_path = "data\\E7_textures"

def ramp_fn(ramp, photoNumber):
	if ramp == 'ramp_1':
		if photoNumber < 62:
			return (photoNumber-5)*0.3 + 40
		else:
			return (photoNumber-62)*0.05 + 57	
	elif ramp == 'rampa_4':
		if photoNumber < 227:
			return (photoNumber-177)*0.3 + 40
		else:
			return (photoNumber-227)*0.05 + 55		


def get_samples(data_path, ramps=('ramp_1', 'rampa_4')):
	samples=[]
	for root, dirs, files in os.walk(data_path):
		for file in files:
			filepath = f"{root}\\{file}"
			ramp1, ramp2 = ramps
			if ramp1 in filepath:
				go = None
				try:
					img = Image.open(filepath)
					go = True 
				except:
					go = False
				if go:
					photoNumber = int(file.split('.')[0])
					temp = ramp_fn(ramp1, photoNumber)
					samples.append([img, temp])
			elif ramp2 in filepath:
				go = None
				try:
					img = Image.open(filepath)
					go = True 
				except:
					go = False
				if go:
					photoNumber = int(file.split('.')[0])
					temp = ramp_fn(ramp2, photoNumber)
					samples.append([img, temp])				
	return samples


def r2_score(y_pred, y_true):
	ss_res = torch.sum((y_true - y_pred) ** 2)
	ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
	r2 = 1 - ss_res / ss_tot
	return r2.item()

def graph_fn(values, title, ylabel, save_path):
	epochs = range(len(values))
	plt.figure(figsize=(8, 6))
	plt.plot(epochs, values, marker='o', linestyle='-', color='b', label=ylabel)
	plt.xlabel('Epochs')
	plt.ylabel(ylabel)
	plt.title(title)
	plt.legend()
	plt.grid(True)
	plt.savefig(save_path)

def target_helper_function(target):
	return target.float().unsqueeze(dim=1)

transform = transforms.Compose([
	transforms.Resize(config['img_shape'][1:]),
	transforms.ToTensor(),
	])

samples = get_samples(data_path)
print(f"Len Samples: {len(samples)}")
dataset = DataUtils.CustomDataset(samples, transforms=transform)


net = Net(config=config).to(config['device'])
trainer = TrainUtils.Trainer(config=config, net=net, target_helper_function=target_helper_function, score_helper_function=r2_score)
trainer.cross_validation_train(dataset, n_splits=5, graph_fn=graph_fn)