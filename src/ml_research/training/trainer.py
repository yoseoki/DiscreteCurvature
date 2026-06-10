import sys, os
os.path.abspath(os.path.dirname(__file__))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import random as rd
import numpy as np
from ml_research.models import build_model, get_non_save_layers, get_layer_groups
from ml_research.optim import buffer as buf
from ml_research.optim import sam
import matplotlib.pyplot as plt
import math
import json
from omegaconf import OmegaConf, DictConfig
from overrides import overrides
from safetensors.torch import save_file, safe_open
import cupy as cp
from ml_research.analysis.pca import WeightPCA
from ml_research.analysis.subspace import SubspaceDiff
import seaborn as sns
from tqdm import tqdm
from random import(shuffle)
from ml_research.data.autoaugment import CIFAR10Policy
from random import shuffle
import time

def smoothing(signal, w_size=3):

	result = []
	totalNum = len(signal)
	offset = w_size // 2
	for i in range(offset, totalNum - offset):
		part = signal[i-1:i+2]
		part.sort()
		result.append(part[offset])
	
	return result

def print_magContainer(magContainer):
	if str(type(magContainer)) == "<class 'numpy.float32'>":
		print("[{:.4f}]".format(magContainer))
	elif len(magContainer) <= 10:
		print("[", end="")
		for i in range(magContainer.shape[0]):
			if i == magContainer.shape[0] - 1: print("{:.4f}]".format(magContainer[i]))
			else: print("{:.4f}".format(magContainer[i]), end=", ")
	else:
		print("[", end="")
		for i in range(5):
			print("{:.4f}".format(magContainer[i]), end=", ")
		print("...", end=", ")
		for i in range(5):
			if i == 4: print("{:.4f}]".format(magContainer[-1]))
			else:
				print("{:.4f}".format(magContainer[i-5]), end=", ")

def grad_scaler_creator(magContainer):
	def grad_scaling(grad):
		scale = torch.ones_like(grad)
		if str(type(magContainer)) == "<class 'numpy.float32'>":
			scale = magContainer * scale
		else:
			for i in range(grad.size(0)):
				scale[i] = magContainer[i] * scale[i]
		result = scale * grad
		return result
	return grad_scaling

class modelLoader:

	def __init__(self, prefix_w, num_classes):
		self.model = None
		self.non_save_layers = None
		self.prefix_w = prefix_w
		self.num_classes = num_classes

	def save_weight(self, is_verbose=False):

		flat = nn.Flatten(start_dim=0)
		counter = 0
		if is_verbose: print()

		for name, param in self.model.named_parameters(): # iteration : each parameter
			
			nonSaveFoundFlag = False
			for element in self.non_save_layers:
				if element in name: nonSaveFoundFlag = True
			if nonSaveFoundFlag: continue

			if is_verbose:
				print("{} : {} | {}".format(counter+1, name, param.shape))

			param_flatten = flat(param)
			param_flatten = param_flatten.detach().tolist()
			counter = counter + 1

			# save each parameter elements in divided csv file
			partitionNum = math.ceil(len(param_flatten) / 1000) 
			for m in range(partitionNum): # iteration : each divied partition in one parameter
				if not os.path.isdir("./" + self.prefix_w + "/layer{}".format(counter)):
					os.mkdir("./" + self.prefix_w + "/layer{}".format(counter))
				f = open("./" + self.prefix_w + "/layer{}/part{:04d}.csv".format(counter, m), 'a')
				if m < partitionNum - 1:
					for n in range(1000):
						if n != 0: f.write(",")
						f.write(str(param_flatten[m * 1000 + n]))
					f.write("\n")
					f.close()
				elif m == partitionNum - 1:
					for n in range(len(param_flatten) - 1000 * m):
						if n != 0: f.write(",")
						f.write(str(param_flatten[m * 1000 + n]))
					f.write("\n")
					f.close()

	def save_weight(self, epoch, is_verbose=False, is_init=False, is_all=False):

		counter = 0
		tensors = {}
		if is_verbose: print()

		for name, param in self.model.named_parameters(): # iteration : each parameter
			
			if not is_all:
				nonSaveFoundFlag = False
				for element in self.non_save_layers:
					if element in name: nonSaveFoundFlag = True
				if nonSaveFoundFlag: continue

			if is_verbose:
				print("{} : {} | {}".format(counter+1, name, param.shape))

			param_copied = param.detach().clone()
			tensors["{:03d}".format(counter+1)] = param_copied
			counter = counter + 1
		if is_all:
			if is_init:
				torch.save(self.model.state_dict(), self.prefix_w + "/weights_epoch_init_all.pt")
			else:
				torch.save(self.model.state_dict(), self.prefix_w + "/weights_epoch{:03d}_all.pt".format(epoch))
		else:
			if is_init: save_file(tensors, self.prefix_w + "/weights_epoch_init.safetensors")
			else: save_file(tensors, self.prefix_w + "/weights_epoch{:03d}.safetensors".format(epoch))

	def save_gradient(self, epoch, is_verbose=False, is_init=False, is_all=False):

		counter = 0
		tensors = {}
		if is_verbose: print()

		for name, param in self.model.named_parameters(): # iteration : each parameter
			
			if not is_all:
				nonSaveFoundFlag = False
				for element in self.non_save_layers:
					if element in name: nonSaveFoundFlag = True
				if nonSaveFoundFlag: continue

			if is_verbose:
				print("{} : {} | {}".format(counter+1, name, param.shape))

			grad_copied = param.grad.detach().clone()
			tensors["{:03d}".format(counter+1)] = grad_copied
			counter = counter + 1
		if is_all:
			if is_init: save_file(tensors, self.prefix_w + "/gradients_epoch_init_all.safetensors")
			else: save_file(tensors, self.prefix_w + "/gradients_epoch{:03d}_all.safetensors".format(epoch))
		else:
			if is_init: save_file(tensors, self.prefix_w + "/gradients_epoch_init.safetensors")
			else: save_file(tensors, self.prefix_w + "/gradients_epoch{:03d}.safetensors".format(epoch))

# for class which inherit trainer class(ex. CIFAR10Trainer, MNISTTrainer, ...)
# need to implement load_DB method, build_num_classes method
# and can train by training method
class trainer:

	def __init__(self):
		
		self.seed = 0
		self.model_name = None
		self.prefix_w = None
		self.lr = 0.0
		self.epochs = 0
		self.batch_size = 0
		self.sampling_step = [0]
		self.model_loader = None
		self.optimizer = None
		self.criterion = None
		self.config = None
		self.color_list = ["blue", "orange", "red", "purple", "green", 
             "olive", "brown", "grey", "cyan", "pink",
             "navy", "lime", "black", "yellow", "crimson",
             "gold", "skyblue", "indigo", "darkgreen", "ivory",
             "blue", "orange", "red", "purple", "green", 
             "olive", "brown", "grey", "cyan", "pink",
             "navy", "lime", "black", "yellow", "crimson",
             "gold", "skyblue", "indigo", "darkgreen", "ivory",
             "blue", "orange", "red", "purple", "green", 
             "olive", "brown", "grey", "cyan", "pink",
             "navy", "lime", "black", "yellow", "crimson",
             "gold", "skyblue", "indigo", "darkgreen", "ivory"]

		self.train_cost_container = []
		self.train_acc_container = []
		self.test_cost_container = []
		self.test_acc_container = []
		self.time_container = []

		self.num_classes = 0
		self.build_num_classes()

	def set_seed(self):
		torch.manual_seed(self.seed)
		rd.seed(self.seed)

	def prepare_save_folder(self):

		from hydra.core.hydra_config import HydraConfig
		self.prefix_w = HydraConfig.get().runtime.output_dir
		os.makedirs(self.prefix_w, exist_ok=True)

	def parse_training_args(self, cfg):

		self.config = OmegaConf.to_container(cfg, resolve=True)

		# 0. meta args
		self.seed = int(cfg.seed)
		self.save_flag = cfg.get("save_flag", False)
		self.save_step = list(cfg.save_step)

		# 1. model args
		self.model_name = cfg.model.model_name
		self.model_backend = cfg.model.get("model_backend", "timm")
		
		# 2. basic parameter args
		self.lr = float(cfg.hparams.learning_rate)
		self.epochs = int(cfg.epochs)
		self.batch_size = int(cfg.hparams.batch_size)
		self.criterion = cfg.criterion # options : "CE"

		# 3. optimizer-specific args
		self.optimizer_name = cfg.optimization.optimizer_name # options : "SGD", "ADAM", "ADAMW", "proposed"

		if self.optimizer_name == "proposed":
			# i  . reshaping
			self.reshape_mode = cfg.hparams.get("reshape_mode", "in_dim")
			# ii . indicator
			self.mag_mode = cfg.hparams.get("mag_mode", "orth")
			# iii. salting
			self.salt_policy = cfg.hparams.get("salt_policy", "none") # options : "none", "direct", "xavier"
			self.square_flag = cfg.hparams.get("square_flag", False)
			# iv . frequency
			self.calc_policy = cfg.hparams.get("calc_policy", "epoch") # options : "epoch", "step"
			self.epoch_offset = int(cfg.hparams.get("epoch_offset", 5))
			self.epoch_interval = int(cfg.hparams.get("epoch_interval", 5))
			self.step_interval = int(cfg.hparams.get("step_interval", 1))
			self.sampling_step = list(cfg.hparams.get("sampling_step"))
			# v  . moving average
			self.mean_beta = float(cfg.hparams.get("mean_beta", 1.0))
			# vi . layer-level processing
			self.layer_process = cfg.hparams.get("layer_process", "normal")
			# vii. weight decay
			self.weight_decay = float(cfg.hparams.get("weight_decay", 0.0))
		elif self.optimizer_name == "SGD" or self.optimizer_name == "SGDW":
			self.weight_decay = float(cfg.hparams.get("weight_decay", 0.0))
			self.schedule = cfg.hparams.get("schedule", None)
		elif self.optimizer_name == "ADAM":
			self.schedule = cfg.hparams.get("schedule", None)
		elif self.optimizer_name == "ADAMW":
			self.weight_decay = float(cfg.hparams.get("weight_decay", 0.0))
			self.schedule = cfg.hparams.get("schedule", None)
		elif self.optimizer_name == "SAM":
			self.weight_decay = float(cfg.hparams.get("weight_decay", 0.0))
			self.schedule = cfg.hparams.get("schedule", None)
	
	def load_model(self):
		self.model_loader = modelLoader(self.prefix_w, self.num_classes)
		self.model_loader.model = build_model(self.model_name, self.num_classes, backend=self.model_backend)
		self.model_loader.non_save_layers = get_non_save_layers(self.model_name, backend=self.model_backend)
		return self.model_loader.model

	def load_DB(self):
		pass

	def build_num_classes(self):
		pass

	def load_optimizer(self, model):
		if self.optimizer_name == "SGD" or self.optimizer_name == "SGDW":
			return torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
		elif self.optimizer_name == "ADAM":
			return torch.optim.Adam(model.parameters(), lr=self.lr)
		elif self.optimizer_name == "ADAMW":
			return torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
		elif self.optimizer_name == "SAM":
			base_optimzier = torch.optim.AdamW
			return sam.SAM(model.parameters(), base_optimzier, lr=self.lr, weight_decay=self.weight_decay)
	
	def load_layerwise_optimizer(self, model):
		names, param_groups = get_layer_groups(self.model_name, model, backend=self.model_backend)
		opt_params = [{'params': pg, 'lr': self.lr, 'momentum': 0.9} for pg in param_groups]
		return names, torch.optim.SGD(opt_params, weight_decay=self.weight_decay)
			
	def update_optimizer(self, optimizer, magContainer):
		
		for i, param_group in enumerate(optimizer.param_groups):
			param_group['lr'] = self.lr * magContainer[i]


	def load_criterion(self):
		if self.criterion == "CE":
			return nn.CrossEntropyLoss()

	def save_training_result(self):

		import csv
		from omegaconf import OmegaConf

		# save config as YAML
		cfg_obj = OmegaConf.create(self.config)
		with open(os.path.join(self.prefix_w, "config.yaml"), 'w') as f:
			OmegaConf.save(config=cfg_obj, f=f)

		# save wallclock time
		with open(os.path.join(self.prefix_w, "time.csv"), 'w', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(["epoch", "walltime-per-epoch"])
			for i, t in enumerate(self.time_container, start=1):
				writer.writerow([i, t])

		# save individual metric CSVs with headers
		individual_metrics = [
			("train_cost.csv", "train_cost", self.train_cost_container),
			("train_acc.csv",  "train_acc",  self.train_acc_container),
			("test_cost.csv",  "test_cost",  self.test_cost_container),
			("test_acc.csv",   "test_acc",   self.test_acc_container),
		]
		for filename, col, values in individual_metrics:
			with open(os.path.join(self.prefix_w, filename), 'w', newline='') as f:
				writer = csv.writer(f)
				writer.writerow(["epoch", col])
				for i, v in enumerate(values, start=1):
					writer.writerow([i, v])

		# save unified metrics CSV (all four series aligned by epoch)
		with open(os.path.join(self.prefix_w, "metrics.csv"), 'w', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(["epoch", "train_cost", "train_acc", "test_cost", "test_acc"])
			rows = zip(
				self.train_cost_container,
				self.train_acc_container,
				self.test_cost_container,
				self.test_acc_container,
			)
			for i, (tc, ta, vc, va) in enumerate(rows, start=1):
				writer.writerow([i, tc, ta, vc, va])

	def _gradient_projection(self, model_name, model, finalMagContainer, epoch, i):
		_, param_groups = get_layer_groups(model_name, model, backend=self.model_backend)
		printFlag = False
		for group_idx, pg in enumerate(param_groups):
			for p in pg:
				if p.grad is None:
					continue

				w = p.view(-1)
				g = p.grad.view(-1)
				w_norm_sq = torch.dot(w, w)
				if w_norm_sq == 0:
					continue

				if group_idx == 0 and epoch == self.offset and i == 0 and not printFlag:
					printFlag = True
					print("Currently directional gradient is magnified!")

				projected_grad = (torch.dot(g, w) / w_norm_sq) * w
				g_orth = g - projected_grad
				g_orth.mul_(finalMagContainer[group_idx])
				p.grad.copy_(g_orth.view_as(p))

	def training(self, cfg, is_verbose=False):
		self.parse_training_args(cfg)
		self.set_seed()
		self.prepare_save_folder()
		trainloader, testloader = self.load_DB()

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		model = self.load_model().to(device)

		if is_verbose:
			a = 1
			for name, param in model.named_parameters():
				print("{} : {} | {}".format(a, name, param.shape))
				a += 1

		criterion = self.load_criterion()

		if self.optimizer_name == "proposed":
			save_name, optimizer = self.load_layerwise_optimizer(model)
			if is_verbose:
				print("save_name : ", save_name)
				print("non_save_layers : ", self.model_loader.non_save_layers)
			obb = buf.OrthBasisBuffer(model, save_name, self.model_loader.non_save_layers, self.square_flag, self.salt_policy, reshape_mode=self.reshape_mode, mag_mode=self.mag_mode)
			obb.update()
			scheduler = None
		else:
			optimizer = self.load_optimizer(model)
			if self.schedule == "COS":
				scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=0)
			elif self.schedule == "CYCLE":
				if "ADAM" in self.optimizer_name:
					scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, steps_per_epoch=len(trainloader), epochs=self.epochs, pct_start=0.2, div_factor=10, final_div_factor=1e3, anneal_strategy="cos", cycle_momentum=False)
				else:
					scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, steps_per_epoch=len(trainloader), epochs=self.epochs, pct_start=0.3, div_factor=25, final_div_factor=1e4, anneal_strategy="cos", cycle_momentum=True, base_momentum=0.85, max_momentum=0.95)
			else:
				scheduler = None
		
		# save point
		if self.save_flag:
			self.model_loader.save_weight(-1, is_verbose=is_verbose, is_init=True)
		
		torch.cuda.synchronize()
		start = time.perf_counter()
		print("========== training start! ==========")

		# save magnitude change
		totalBaseContainer = []
		totalSaltedContainer = []
		baseMagValues = None

		for epoch in range(self.epochs):
			torch.cuda.synchronize()
			epoch_start = time.perf_counter()
			print("epoch : {:03d} / {:03d} - ".format(epoch + 1, self.epochs), end="")

			if self.optimizer_name == "proposed":
				if self.calc_policy == "epoch":
					if epoch == self.epoch_offset - 1:
						obb.set_basis()
						baseMagValues = obb.get_base_magnitude()
					if epoch > self.epoch_offset - 1 and (epoch - self.epoch_offset) % self.epoch_interval == 0:

						# basic part for calculating magnitude
						magContainer = obb.calc_magnitude()
						totalBaseContainer.append(magContainer)

						alphas = obb.getAplhas(self.model_name)

						# EMA
						if len(totalSaltedContainer) == 0:
							finalMagContainer = [alphas[lIndex] * magContainer[lIndex] for lIndex in range(len(magContainer))]
						else:
							finalMagContainer = totalSaltedContainer[-1].copy()
							for lIndex in range(len(magContainer)):
								finalMagContainer[lIndex] = (1 - self.mean_beta)*finalMagContainer[lIndex] + self.mean_beta*(magContainer[lIndex] * alphas[lIndex])

						# layer process (normal / mean / shuffle)
						if self.layer_process == "mean":
							average_mag = sum(finalMagContainer) / len(finalMagContainer)
							for i in range(len(finalMagContainer)): finalMagContainer[i] = average_mag
						elif self.layer_process == "shuffle":
							shuffle(finalMagContainer)

						print()
						print("orig coef : ", end="")
						for element in finalMagContainer: print("{:.4f}".format(element), end=" || ")
						print()

						totalSaltedContainer.append(finalMagContainer)
						if self.weight_decay == 0.0:
							self.update_optimizer(optimizer, finalMagContainer)

						if all(x <= 0.05 for x in magContainer): break

				# Just preparing... real sampling and update are implemented in inner loop
				elif self.calc_policy == "step":
					if epoch == self.epoch_offset - 1 or (epoch > self.epoch_offset - 1 and (epoch - self.epoch_offset) % self.epoch_interval == 0):
						mag_buffer = []
						for _ in range(len(save_name)): mag_buffer.append([])
						obb.clear_buffer()

			model.train()
			avg_cost = 0
			train_correct = 0
			train_total = 0

			for i, data in enumerate(trainloader):

				X, Y = data

				X = X.to(device)
				Y = Y.to(device)

				if self.optimizer_name == "SAM":
					# --- 1st pass: 현재 점 w에서 gradient 계산 → first_step이 w+e(w)로 climb ---
					prediction = model(X)
					cost = criterion(prediction, Y)
					cost.backward()
					optimizer.first_step(zero_grad=True)

					# --- 2nd pass: perturbed 점에서 다시 forward + backward → second_step이 실제 업데이트 ---
					criterion(model(X), Y).backward()   # 반드시 전체 forward를 다시
					optimizer.second_step(zero_grad=True)

				else:
					optimizer.zero_grad()
					prediction = model(X)
					cost = criterion(prediction, Y)
					cost.backward()

					if self.optimizer_name == "proposed" and self.weight_decay != 0.0:
						with torch.no_grad():
							if "finalMagContainer" in locals():
								self._gradient_projection(self.model_name, model, finalMagContainer, epoch, i)

					optimizer.step()

				if scheduler:
					if self.schedule == "CYCLE":
						scheduler.step()
				avg_cost += cost.item()

				with torch.no_grad():
					_, predicted_train = torch.max(prediction, 1)
					train_correct += (predicted_train == Y).sum().item()
					train_total   += Y.size(0)

				if self.optimizer_name == "proposed":
					if self.calc_policy == "step":
						if epoch == self.epoch_offset - 1 and i % self.step_interval == 0:
							obb.update()
							if i > 2 * self.step_interval:
								magContainer = obb.calc_magnitude()
								for j, mag in enumerate(magContainer): mag_buffer[j].append(mag)

						if epoch > self.epoch_offset - 1 and (epoch - self.epoch_offset) % self.epoch_interval == 0 and i % self.step_interval == 0:
							obb.update()
							if i > 2 * self.step_interval:
								magContainer = obb.calc_magnitude()
								for j, mag in enumerate(magContainer): mag_buffer[j].append(mag)

				if i in self.save_step:
					# save point
					if self.save_flag:
						if epoch > self.epoch_offset - 1 and (epoch - self.epoch_offset) % self.epoch_interval == 0:
							self.model_loader.save_weight(len(self.save_step)*epoch + self.save_step.index(i), is_verbose=is_verbose, is_all=True)
							self.model_loader.save_gradient(len(self.save_step)*epoch + self.save_step.index(i), is_verbose=is_verbose, is_all=True)
					
				if i in self.sampling_step:	
					if self.optimizer_name == "proposed":
						if self.calc_policy == "epoch":
							obb.update()
			
			if self.optimizer_name == "proposed":
				if self.calc_policy == "step":
					if epoch == self.epoch_offset - 1:
						_finalMagContainer = []
						for comp in mag_buffer:
							comp_smoothened = smoothing(comp)
							_finalMagContainer.append(sum(comp_smoothened))
						print("base: ", end="")
						for element in _finalMagContainer: print("{:.4f}".format(element), end=" || ")
						print()
						obb.set_basis_manually(_finalMagContainer)
						baseMagValues = obb.get_base_magnitude()
					if epoch > self.epoch_offset - 1 and (epoch - self.epoch_offset) % self.epoch_interval == 0:
						finalMagContainer = []
						alphas = obb.getAplhas(self.model_name)
						tmp = []
						for comp in mag_buffer:
							comp_smoothened = smoothing(comp)
							tmp.append(sum(comp_smoothened))
						totalBaseContainer.append(tmp)
						for alpha, comp in zip(alphas, tmp):
							finalMagContainer.append(comp * alpha)
						for element in finalMagContainer: print("{:.4f}".format(element), end=" || ")
						print()
						totalSaltedContainer.append(finalMagContainer)
						self.update_optimizer(optimizer, finalMagContainer)

			
			avg_cost = avg_cost / len(trainloader)
			self.train_cost_container.append(avg_cost)
			train_acc = train_correct / train_total
			self.train_acc_container.append(train_acc)
			print('train cost = {:>.6f} | train acc = {:>.4f}%'.format(avg_cost, 100 * train_acc), end=" || ")

			with torch.no_grad():
				model.eval()
				correct = 0
				total = 0
				test_cost = 0.0
				for data in testloader:
					images, labels = data
					images = images.to(device)
					labels = labels.to(device)
					outputs = model(images)
					test_cost += criterion(outputs, labels).item()

					# size of outputs : (batch_size, num_classes(=10))
					# torch.max finds the index with maximum value in each data of batch
					# ex. [0.01, 0.01, 0.01, ... 0.91, 0.01] -> 8(index of 0.91)
					_, predicted = torch.max(outputs, 1)
					c = (predicted == labels).squeeze() # [8, 1, 4, 3] vs [8, 2, 4, 4] -> [1, 0, 1, 0]
					for j in range(c.size(dim=0)): # [1, 0, 1, 0] -> correct += 2, total += 4
						correct += c[j].item()
						total += 1
				test_cost = test_cost / len(testloader)
				self.test_cost_container.append(test_cost)
				self.test_acc_container.append(correct / total)
				print('test cost = {:>.6f} | test acc = {:>.4f}%'.format(test_cost, 100 * correct / total))

			# if "finalMagContainer" in locals():
			# 	learningStopFlag = True
			# 	for mag in finalMagContainer:
			# 		if mag > 0.01: learningStopFlag = False
			# 	if learningStopFlag: break
			
			if scheduler:
				if self.schedule == "COS":
					scheduler.step()

			torch.cuda.synchronize()
			epoch_end = time.perf_counter()
			self.time_container.append(epoch_end - epoch_start)

		print("========== training over! ==========")

		torch.cuda.synchronize()
		end = time.perf_counter()
		self.time_container.append(end - start)

		# save model
		torch.save(model.state_dict(), os.path.join(self.prefix_w, "model.pt"))

		self.save_training_result()

		# save magnitude CSVs with layer-name headers
		if self.optimizer_name == "proposed":
			import csv
			mag_header = ["epoch"] + list(save_name)

			if totalBaseContainer:
				with open(os.path.join(self.prefix_w, "mag.csv"), 'w', newline='') as f:
					writer = csv.writer(f)
					writer.writerow(mag_header)
					for i, row in enumerate(totalBaseContainer):
						writer.writerow([self.epoch_offset + 1 + i * self.epoch_interval] + list(row))

			if totalSaltedContainer:
				with open(os.path.join(self.prefix_w, "salted_mag.csv"), 'w', newline='') as f:
					writer = csv.writer(f)
					writer.writerow(mag_header)
					for i, row in enumerate(totalSaltedContainer):
						writer.writerow([self.epoch_offset + 1 + i * self.epoch_interval] + list(row))

			if baseMagValues is not None:
				base_header = list(save_name)
				with open(os.path.join(self.prefix_w, "base_mag.csv"), 'w', newline='') as f:
					writer = csv.writer(f)
					writer.writerow(base_header)
					writer.writerow(list(baseMagValues))

	def validate_geodesic(self, model_name, seed, config_file_path, weight_path, gradient_path, alpha, is_verbose=False, prefix=""):
		self.test_acc_container = []
		self.set_seed(rdSeed)
		self.parse_training_args(config_file_path)
		_, testloader = self.load_DB()
		self.non_save_layers = ["bias", "downsample.1", "bn"]

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		model = self.load_model(model_name).to(device)

		if is_verbose:
			a = 1
			for name, param in model.named_parameters():
				print("{} : {} | {}".format(a, name, param.shape))
				a += 1

		f_grad = safe_open(gradient_path, framework="pt", device="cuda" if torch.cuda.is_available() else "cpu")

		names = ["conv1", "layer11.conv2", "layer21.conv2", "layer31.conv2", "layer41.conv2", "fc"]
		names_alpha = {}
		for i in range(len(alpha)):
			names_alpha[names[i]] = alpha[i]

		for _t in tqdm(np.linspace(0, 5, 50)):

			model.load_state_dict(torch.load(weight_path, weights_only=True))
			counter = 0
			with torch.no_grad():
				for name, param in model.named_parameters(): # iteration : each parameter

					for ele in names:
						if ele in name:
							tmp = 1 + _t*(names_alpha[ele] - 1)
							t = tmp * self.lr
							break

					counter += 1
					grad = f_grad.get_tensor("{:03d}".format(counter))

					param.add_(-t, grad)

				model.eval()
				correct = 0
				total = 0
				for data in testloader:
					images, labels = data
					images = images.to(device)
					labels = labels.to(device)
					outputs = model(images)

					_, predicted = torch.max(outputs, 1)
					c = (predicted == labels).squeeze()
					for j in range(c.size(dim=0)):
						correct += c[j].item()
						total += 1
				self.test_acc_container.append(correct / total)
				# print('[t={:.4f}] Valid Accuracy = {:>.9}%'.format(_t, 100 * correct / total))

		max_value = max(self.test_acc_container)
		max_index = self.test_acc_container.index(max_value)
		print(prefix, "max index : t={:.4f} and max value : {:.4f}".format(np.linspace(0, 5, 50)[max_index], max_value))

class CIFAR10Trainer(trainer):

	@overrides
	def load_DB(self):
		transform = transforms.Compose(
		[
		# CIFAR10Policy(),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])

		trainset = torchvision.datasets.CIFAR10(root='DB', train=True,
												download=True, transform=transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
												shuffle=True, num_workers=2)

		testset = torchvision.datasets.CIFAR10(root='DB', train=False,
											download=True, transform=transform)
		testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
												shuffle=True, num_workers=2)
		
		return trainloader, testloader

	@overrides
	def build_num_classes(self):
		self.num_classes = 10 # class number of CIFAR10

class SVHNTrainer(trainer):

	@overrides
	def load_DB(self):
		transform = transforms.Compose(
		[
		# transforms.Resize(224),
		# transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])

		trainset = torchvision.datasets.SVHN(root='DB', split="train",
												download=True, transform=transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
												shuffle=True, num_workers=2)

		testset = torchvision.datasets.SVHN(root='DB', split="test",
											download=True, transform=transform)
		testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
												shuffle=True, num_workers=2)
		
		return trainloader, testloader

	@overrides
	def build_num_classes(self):
		self.num_classes = 10 # class number of SVHN

class CIFAR100Trainer(trainer):

	@overrides
	def load_DB(self):
		transform = transforms.Compose(
		[
		# transforms.Resize(64),
        # CIFAR10Policy(),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])

		trainset = torchvision.datasets.CIFAR100(root='DB', train=True,
												download=True, transform=transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
												shuffle=True, num_workers=2)

		testset = torchvision.datasets.CIFAR100(root='DB', train=False,
											download=True, transform=transform)
		testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
												shuffle=True, num_workers=2)
		
		return trainloader, testloader

	@overrides
	def build_num_classes(self):
		self.num_classes = 100 # class number of CIFAR10