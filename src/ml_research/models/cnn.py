import sys, os
import torch
import torch.nn as nn
import numpy as np
from ml_research.models import lda_layers as lm
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
from typing import Union
import torch.nn.init as init
import math

# structure : conv(3,4) - conv(4,8) - conv(8,16) - fc(1728,itemNum)
# #param : 108+4 - 288+8 - 1152+16 - 1728*itemNum + itemNum
class CNN_ALOI(nn.Module):

	def __init__(self, itemNum):
		super(CNN_ALOI, self).__init__()

		self.pretreat1 = torch.nn.AvgPool2d(2)
		self.pretreat2 = torch.nn.AvgPool2d(2)
		
		self.layer1 = torch.nn.Sequential(
			nn.Conv2d(3, 4, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.MaxPool2d(2))
		self.layer2 = torch.nn.Sequential(
			nn.Conv2d(4, 8, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.MaxPool2d(2))
		self.layer3 = torch.nn.Sequential(
			nn.Conv2d(8, 16, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.MaxPool2d(2))
		self.fc = torch.nn.Linear(16 * 12 * 9, itemNum, bias=True)

	def forward(self, x):
		x = self.pretreat1(x)
		x = self.pretreat2(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x
	
	def save_models(self, pathPrefix):
		torch.save(self.layer1, pathPrefix +"_layer1.pth")
		torch.save(self.layer2, pathPrefix +"_layer2.pth")
		torch.save(self.layer3, pathPrefix +"_layer3.pth")
		torch.save(self.fc, pathPrefix +"_fc.pth")

	def save_models_sd(self, pathPrefix):
		torch.save(self.layer1.state_dict(), pathPrefix +"_layer1_sd.pth")
		torch.save(self.layer2.state_dict(), pathPrefix +"_layer2_sd.pth")
		torch.save(self.layer3.state_dict(), pathPrefix +"_layer3_sd.pth")
		torch.save(self.fc.state_dict(), pathPrefix +"_fc_sd.pth")

	# can load model parameters by only non-sd save file
	def load_models(self, pathPrefix):
		self.layer1 = torch.load(pathPrefix +"_layer1.pth")
		self.layer2 = torch.load(pathPrefix +"_layer2.pth")
		self.layer3 = torch.load(pathPrefix +"_layer3.pth")
		self.fc = torch.load(pathPrefix +"_fc.pth")

# structure : conv(3,4) - conv(4,8) - conv(8,16) - fc(1728,itemNum)
# #param : 108+4 - 288+8 - 1152+16 - 1728*itemNum + itemNum
class CNN_ALOI_LDA(nn.Module):

	def __init__(self, itemNum, apprNumContainer):
		super(CNN_ALOI_LDA, self).__init__()

		self.pretreat1 = torch.nn.AvgPool2d(2)
		self.pretreat2 = torch.nn.AvgPool2d(2)
		
		self.layer1 = torch.nn.Sequential(
			#nn.Conv2d(3, 4, kernel_size=(3,3), padding=(1,1)),
			lm.LDAConv2d(apprNumContainer[0], 3, 4, 3, 1),
			nn.ReLU(),
			nn.MaxPool2d(2))
		self.layer2 = torch.nn.Sequential(
			#nn.Conv2d(4, 8, kernel_size=(3,3), padding=(1,1)),
			lm.LDAConv2d(apprNumContainer[1], 4, 8, 3, 1),
			nn.ReLU(),
			nn.MaxPool2d(2))
		self.layer3 = torch.nn.Sequential(
			#nn.Conv2d(8, 16, kernel_size=(3,3), padding=(1,1)),
			lm.LDAConv2d(apprNumContainer[2], 8, 16, 3, 1),
			nn.ReLU(),
			nn.MaxPool2d(2))
		#self.fc = torch.nn.Linear(32 * 48 * 36, itemNum, bias=True)
		self.fc = lm.LDAFC(16 * 12 * 9, itemNum, apprNumContainer[3])

	def import_basis(self, layerNum, basis_matrix, init_vec=None, initFlag=False):
		if layerNum == 1:
			for i, sublayer in enumerate(self.layer1.modules()):
				if i == 1:
					sublayer.import_basis(basis_matrix, init_vec, initFlag)
		if layerNum == 2:
			for i, sublayer in enumerate(self.layer2.modules()):
				if i == 1:
					sublayer.import_basis(basis_matrix, init_vec, initFlag)

		if layerNum == 3:
			for i, sublayer in enumerate(self.layer3.modules()):
				if i == 1:
					sublayer.import_basis(basis_matrix, init_vec, initFlag)
		if layerNum == 4:
			self.fc.import_basis(basis_matrix, init_vec, initFlag)

	def forward(self, x):
		x = self.pretreat1(x)
		x = self.pretreat2(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x
	
	def save_models(self, pathPrefix):
		torch.save(self.layer1, pathPrefix +"_layer1.pth")
		torch.save(self.layer2, pathPrefix +"_layer2.pth")
		torch.save(self.layer3, pathPrefix +"_layer3.pth")
		torch.save(self.fc, pathPrefix +"_fc.pth")

	def save_models_sd(self, pathPrefix):
		torch.save(self.layer1.state_dict(), pathPrefix +"_layer1_sd.pth")
		torch.save(self.layer2.state_dict(), pathPrefix +"_layer2_sd.pth")
		torch.save(self.layer3.state_dict(), pathPrefix +"_layer3_sd.pth")
		torch.save(self.fc.state_dict(), pathPrefix +"_fc_sd.pth")

	# can load model parameters by only non-sd save file
	def load_models(self, pathPrefix):
		self.layer1 = torch.load(pathPrefix +"_layer1.pth")
		self.layer2 = torch.load(pathPrefix +"_layer2.pth")
		self.layer3 = torch.load(pathPrefix +"_layer3.pth")
		self.fc = torch.load(pathPrefix +"_fc.pth")

# structure : conv(3,4) - conv(4,8) - conv(8,16) - fc(1728,itemNum)
# #param : 108+4 - 288+8 - 1152+16 - 1728*itemNum + itemNum	
class CNN_ALOI_LDA2(nn.Module):

	def __init__(self, itemNum, apprNum):
		super(CNN_ALOI_LDA2, self).__init__()

		self.inputCh = 3
		self.mediumCh1 = 4
		self.mediumCh2 = 8
		self.outputCh = 16
		self.kernelSize = 3
		self.finalImageSize = 12 * 9

		self.conv1WeightDim = (self.inputCh * self.mediumCh1 * self.kernelSize * self.kernelSize)
		self.conv1BiasDim = (self.mediumCh1)
		self.conv2WeightDim = (self.mediumCh1 * self.mediumCh2 * self.kernelSize * self.kernelSize)
		self.conv2BiasDim = (self.mediumCh2)
		self.conv3WeightDim = (self.mediumCh2 * self.outputCh * self.kernelSize * self.kernelSize)
		self.conv3BiasDim = (self.outputCh)
		self.fcWeightDim = self.outputCh * self.finalImageSize * itemNum
		self.fcBiasDim = itemNum
		self.projDim = self.conv1WeightDim + self.conv1BiasDim + self.conv2WeightDim + self.conv2BiasDim + self.conv3WeightDim + self.conv3BiasDim + self.fcWeightDim + self.fcBiasDim
		self.apprDim = apprNum

		self.pretreat1 = torch.nn.AvgPool2d(2)
		self.pretreat2 = torch.nn.AvgPool2d(2)
		
		self.conv1 = lm.LDA2Conv2d(3, 4, 3, 1)
		self.relu1 = nn.ReLU()
		self.pool1 = nn.MaxPool2d(2)

		self.conv2 = lm.LDA2Conv2d(4, 8, 3, 1)
		self.relu2 = nn.ReLU()
		self.pool2 = nn.MaxPool2d(2)

		self.conv3 = lm.LDA2Conv2d(8, 16, 3, 1)
		self.relu3 = nn.ReLU()
		self.pool3 = nn.MaxPool2d(2)

		self.fc = lm.LDA2FC(16 * 12 * 9, itemNum)

		self.proj = torch.randn(self.projDim, apprNum, requires_grad=False)
		self.apprWeight = torch.nn.Parameter(torch.randn(apprNum, 1), requires_grad=True)
		self._parameters["apprWeight"] = self.apprWeight

	def forward(self, x):

		tmp = torch.matmul(self.proj, self.apprWeight)

		tmpIdx1 = self.conv1WeightDim
		tmpIdx2 = tmpIdx1 + self.conv1BiasDim
		self.conv1.set_weight(tmp[:tmpIdx1,:], tmp[tmpIdx1:tmpIdx2,:])
		tmpIdx3 = tmpIdx2 + self.conv2WeightDim
		tmpIdx4 = tmpIdx3 + self.conv2BiasDim
		self.conv2.set_weight(tmp[tmpIdx2:tmpIdx3,:], tmp[tmpIdx3:tmpIdx4,:])
		tmpIdx5 = tmpIdx4 + self.conv3WeightDim
		tmpIdx6 = tmpIdx5 + self.conv3BiasDim
		self.conv3.set_weight(tmp[tmpIdx4:tmpIdx5,:], tmp[tmpIdx5:tmpIdx6,:])
		tmpIdx7 = tmpIdx6 + self.fcWeightDim
		tmpIdx8 = tmpIdx7 + self.fcBiasDim
		self.fc.set_weight(tmp[tmpIdx6:tmpIdx7,:], tmp[tmpIdx7:tmpIdx8,:])

		x = self.pretreat1(x)
		x = self.pretreat2(x)
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.pool1(x)
		x = self.conv2(x)
		x = self.relu2(x)
		x = self.pool2(x)
		x = self.conv3(x)
		x = self.relu3(x)
		x = self.pool3(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x
	
	def import_basis(self, basis_matrix, initFlag):

		if self.apprDim != basis_matrix.shape[1]:
			print("dimension of appoximate space is not equal to number of basis!")
			return
		
		if self.projDim != basis_matrix.shape[0]:
			print("dimension of projection space is not equal to dimension of basis!")
			return
		
		if initFlag:
			tmp = np.linalg.inv(np.transpose(basis_matrix) @ basis_matrix) @ np.transpose(basis_matrix)
			init_vec = np.sqrt(2 / sum(basis_matrix.shape)) * np.random.randn(basis_matrix.shape[0], 1)
			print(init_vec.shape)
			init_vec_appr  = tmp @ init_vec
			print(init_vec_appr.shape)
			
			self.apprWeight = torch.nn.Parameter(torch.FloatTensor(init_vec_appr), requires_grad=True)
		
		self.proj = torch.FloatTensor(basis_matrix)
		self.proj.requires_grad = False
	
	def save_models(self, pathPrefix):
		torch.save(self.apprWeight, pathPrefix +"_appr.pth")

	def save_models_sd(self, pathPrefix):
		torch.save(self.apprWeight.state_dict(), pathPrefix +"_appr_sd.pth")

	# can load model parameters by only non-sd save file
	def load_models(self, pathPrefix):
		self.apprWeight = torch.load(pathPrefix +"_appr.pth")

# structure : conv(1,8) - conv(8,16) - fc(784,10)
# #param : 72+8 - 1152+16 - 7840+10
class CNN_MNIST(nn.Module):

	def __init__(self, itemNum):
		super(CNN_MNIST, self).__init__()
		
		self.layer1 = torch.nn.Sequential(
			nn.Conv2d(1, 2, kernel_size=(3,3), padding=(1,1), bias=False),
			nn.ReLU(),
			nn.MaxPool2d(2))
		self.layer2 = torch.nn.Sequential(
			nn.Conv2d(2, 2, kernel_size=(3,3), padding=(1,1), bias=False),
			nn.ReLU(),
			nn.MaxPool2d(2))
		self.layer3 = torch.nn.Sequential(
			nn.Conv2d(2, 4, kernel_size=(3,3), padding=(1,1), bias=False),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2))
		self.fc = torch.nn.Linear(4 * 3 * 3, itemNum, bias=False)

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x
	
	def save_models(self, pathPrefix):
		torch.save(self.layer1, pathPrefix +"_layer1.pth")
		torch.save(self.layer2, pathPrefix +"_layer2.pth")
		torch.save(self.layer3, pathPrefix +"_layer3.pth")
		torch.save(self.fc, pathPrefix +"_fc.pth")

	def save_models_sd(self, pathPrefix):
		torch.save(self.layer1.state_dict(), pathPrefix +"_layer1_sd.pth")
		torch.save(self.layer2.state_dict(), pathPrefix +"_layer2_sd.pth")
		torch.save(self.layer3.state_dict(), pathPrefix +"_layer3_sd.pth")
		torch.save(self.fc.state_dict(), pathPrefix +"_fc_sd.pth")

	# can load model parameters by only non-sd save file
	def load_models(self, pathPrefix):
		self.layer1 = torch.load(pathPrefix +"_layer1.pth")
		self.layer2 = torch.load(pathPrefix +"_layer2.pth")
		self.layer3 = torch.load(pathPrefix +"_layer3.pth")
		self.fc = torch.load(pathPrefix +"_fc.pth")

# structure : conv(1,8) - conv(8,16) - fc(784,10)
# #param : 72+8 - 1152+16 - 7840+10
class CNN_MNIST_LDA(nn.Module):

	def __init__(self, itemNum, apprNumContainer):
		super(CNN_MNIST_LDA, self).__init__()
		
		self.layer1 = torch.nn.Sequential(
			#nn.Conv2d(1, 8, kernel_size=(3,3), padding=(1,1)),
			lm.LDAConv2d(apprNumContainer[0], 1, 8, 3, 1),
			nn.ReLU(),
			nn.MaxPool2d(2))
		self.layer2 = torch.nn.Sequential(
			#nn.Conv2d(8, 16, kernel_size=(3,3), padding=(1,1)),
			lm.LDAConv2d(apprNumContainer[1], 8, 16, 3, 1),
			nn.ReLU(),
			nn.MaxPool2d(2))
		self.fc = lm.LDAFC(16 * 7 * 7, itemNum, apprNumContainer[2])

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x
	
	def import_basis(self, layerNum, basis_matrix):
		if layerNum == 1:
			for i, sublayer in enumerate(self.layer1.modules()):
				if i == 1:
					sublayer.import_basis(basis_matrix)
				else: pass
		if layerNum == 2:
			for i, sublayer in enumerate(self.layer2.modules()):
				if i == 1:
					sublayer.import_basis(basis_matrix)
				else: pass
		if layerNum == 3:
			self.fc.import_basis(basis_matrix)
	
	def save_models(self, pathPrefix):
		torch.save(self.layer1, pathPrefix +"_layer1.pth")
		torch.save(self.layer2, pathPrefix +"_layer2.pth")
		torch.save(self.fc, pathPrefix +"_fc.pth")

	def save_models_sd(self, pathPrefix):
		torch.save(self.layer1.state_dict(), pathPrefix +"_layer1_sd.pth")
		torch.save(self.layer2.state_dict(), pathPrefix +"_layer2_sd.pth")
		torch.save(self.fc.state_dict(), pathPrefix +"_fc_sd.pth")

	# can load model parameters by only non-sd save file
	def load_models(self, pathPrefix):
		self.layer1 = torch.load(pathPrefix +"_layer1.pth")
		self.layer2 = torch.load(pathPrefix +"_layer2.pth")
		self.fc = torch.load(pathPrefix +"_fc.pth")

# structure : conv(1,8) - conv(8,16) - fc(784,10)
# #param : 72+8 - 1152+16 - 7840+10
class CNN_MNIST_LDA2(nn.Module):

	def __init__(self, itemNum, apprNum):
		super(CNN_MNIST_LDA2, self).__init__()

		self.inputCh = 1
		self.mediumCh1 = 8
		self.outputCh = 16
		self.kernelSize = 3
		self.finalImageSize = 7 * 7

		self.conv1WeightDim = (self.inputCh * self.mediumCh1 * self.kernelSize * self.kernelSize)
		self.conv1BiasDim = (self.mediumCh1)
		self.conv2WeightDim = (self.mediumCh1 * self.outputCh * self.kernelSize * self.kernelSize)
		self.conv2BiasDim = (self.outputCh)
		self.fcWeightDim = self.outputCh * self.finalImageSize * itemNum
		self.fcBiasDim = itemNum
		self.projDim = self.conv1WeightDim + self.conv1BiasDim + self.conv2WeightDim + self.conv2BiasDim + self.fcWeightDim + self.fcBiasDim
		self.apprDim = apprNum
		
		self.conv1 = lm.LDA2Conv2d(1, 8, 3, 1) #nn.Conv2d(1, 8, kernel_size=(3,3), padding=(1,1))
		self.relu1 = nn.ReLU()
		self.pool1 = nn.MaxPool2d(2)

		self.conv2 = lm.LDA2Conv2d(8, 16, 3, 1) #nn.Conv2d(8, 16, kernel_size=(3,3), padding=(1,1))
		self.relu2 = nn.ReLU()
		self.pool2 = nn.MaxPool2d(2)

		self.fc = lm.LDA2FC(16 * 7 * 7, itemNum)

		self.proj = torch.randn(self.projDim, apprNum, requires_grad=False)
		self.apprWeight = torch.nn.Parameter(torch.randn(apprNum, 1), requires_grad=True)
		self._parameters["apprWeight"] = self.apprWeight

	def forward(self, x):

		tmp = torch.matmul(self.proj, self.apprWeight)

		tmpIdx1 = self.conv1WeightDim
		tmpIdx2 = tmpIdx1 + self.conv1BiasDim
		self.conv1.set_weight(tmp[:tmpIdx1,:], tmp[tmpIdx1:tmpIdx2,:])
		tmpIdx3 = tmpIdx2 + self.conv2WeightDim
		tmpIdx4 = tmpIdx3 + self.conv2BiasDim
		self.conv2.set_weight(tmp[tmpIdx2:tmpIdx3,:], tmp[tmpIdx3:tmpIdx4,:])
		tmpIdx5 = tmpIdx4 + self.fcWeightDim
		tmpIdx6 = tmpIdx5 + self.fcBiasDim
		self.fc.set_weight(tmp[tmpIdx4:tmpIdx5,:], tmp[tmpIdx5:tmpIdx6,:])

		x = self.conv1(x)
		x = self.relu1(x)
		x = self.pool1(x)
		x = self.conv2(x)
		x = self.relu2(x)
		x = self.pool2(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x
	
	def import_basis(self, basis_matrix, initFlag):

		if self.apprDim != basis_matrix.shape[1]:
			print("dimension of appoximate space is not equal to number of basis!")
			return
		
		if self.projDim != basis_matrix.shape[0]:
			print("dimension of projection space is not equal to dimension of basis!")
			return
		
		if initFlag:
			tmp = np.linalg.inv(np.transpose(basis_matrix) @ basis_matrix) @ np.transpose(basis_matrix)
			init_vec = np.sqrt(2 / sum(basis_matrix.shape)) * np.random.randn(basis_matrix.shape[0], 1)
			init_vec_appr  = tmp @ init_vec
			
			self.apprWeight = torch.nn.Parameter(torch.FloatTensor(init_vec_appr), requires_grad=True)
		
		self.proj = torch.FloatTensor(basis_matrix)
		self.proj.requires_grad = False

	def save_models(self, pathPrefix):
		torch.save(self.apprWeight, pathPrefix +"_appr.pth")

	def save_models_sd(self, pathPrefix):
		torch.save(self.apprWeight.state_dict(), pathPrefix +"_appr_sd.pth")

	# can load model parameters by only non-sd save file
	def load_models(self, pathPrefix):
		self.apprWeight = torch.load(pathPrefix +"_appr.pth")


# structure : conv(3,8) - conv(8,16) - conv(16,32) - conv(32,64) - fc(256,10)
# #param : 216+8 - 1152+16 - 4608+32 - 18432+64 - 2560+10 = 27098
class CNN_CIFAR(nn.Module):

	def __init__(self):
		super(CNN_CIFAR, self).__init__()
		
		self.layer1 = torch.nn.Sequential(
			nn.Conv2d(3, 8, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.MaxPool2d(2))
		self.layer2 = torch.nn.Sequential(
			nn.Conv2d(8, 16, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.MaxPool2d(2))
		self.layer3 = torch.nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.MaxPool2d(2))
		self.layer4 = torch.nn.Sequential(
			nn.Conv2d(32, 64, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.MaxPool2d(2))
		self.fc = torch.nn.Linear(64 * 2 * 2, 10, bias=True)

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x
	
	def save_models(self, pathPrefix):
		torch.save(self.layer1, pathPrefix +"_layer1.pth")
		torch.save(self.layer2, pathPrefix +"_layer2.pth")
		torch.save(self.layer3, pathPrefix +"_layer3.pth")
		torch.save(self.layer4, pathPrefix +"_layer4.pth")
		torch.save(self.fc, pathPrefix +"_fc.pth")

	def save_models_sd(self, pathPrefix):
		torch.save(self.layer1.state_dict(), pathPrefix +"_layer1_sd.pth")
		torch.save(self.layer2.state_dict(), pathPrefix +"_layer2_sd.pth")
		torch.save(self.layer3.state_dict(), pathPrefix +"_layer3_sd.pth")
		torch.save(self.layer4.state_dict(), pathPrefix +"_layer4_sd.pth")
		torch.save(self.fc.state_dict(), pathPrefix +"_fc_sd.pth")

	# can load model parameters by only non-sd save file
	def load_models(self, pathPrefix):
		self.layer1 = torch.load(pathPrefix +"_layer1.pth")
		self.layer2 = torch.load(pathPrefix +"_layer2.pth")
		self.layer3 = torch.load(pathPrefix +"_layer3.pth")
		self.layer4 = torch.load(pathPrefix +"_layer4.pth")
		self.fc = torch.load(pathPrefix +"_fc.pth")


class UNET(nn.Module):
		
	def __init__(self, num_classes):
		super(UNET, self).__init__()
		self.num_classes = num_classes
		self.contracting_11 = self.conv_block(in_channels=3, out_channels=32)
		self.contracting_12 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.contracting_21 = self.conv_block(in_channels=32, out_channels=64)
		self.contracting_22 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.contracting_31 = self.conv_block(in_channels=64, out_channels=128)
		self.contracting_32 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.contracting_41 = self.conv_block(in_channels=128, out_channels=256)
		self.contracting_42 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.middle = self.conv_block(in_channels=256, out_channels=512)
		self.expansive_11 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
		self.expansive_12 = self.conv_block(in_channels=512, out_channels=256)
		self.expansive_21 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
		self.expansive_22 = self.conv_block(in_channels=256, out_channels=128)
		self.expansive_31 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
		self.expansive_32 = self.conv_block(in_channels=128, out_channels=64)
		self.expansive_41 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
		self.expansive_42 = self.conv_block(in_channels=64, out_channels=32)
		self.fc = torch.nn.Linear(32 * 32 * 32, num_classes, bias=True)
		
	def conv_block(self, in_channels, out_channels):
		block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels),
                                    nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels))
		return block
	
	def forward(self, X):
		contracting_11_out = self.contracting_11(X) # [-1, 64, 256, 256]
		contracting_12_out = self.contracting_12(contracting_11_out) # [-1, 64, 128, 128]
		contracting_21_out = self.contracting_21(contracting_12_out) # [-1, 128, 128, 128]
		contracting_22_out = self.contracting_22(contracting_21_out) # [-1, 128, 64, 64]
		contracting_31_out = self.contracting_31(contracting_22_out) # [-1, 256, 64, 64]
		contracting_32_out = self.contracting_32(contracting_31_out) # [-1, 256, 32, 32]
		contracting_41_out = self.contracting_41(contracting_32_out)
		contracting_42_out = self.contracting_42(contracting_41_out)
		middle_out = self.middle(contracting_42_out) # [-1, 1024, 16, 16]
		expansive_11_out = self.expansive_11(middle_out) # [-1, 512, 32, 32]
		expansive_12_out = self.expansive_12(torch.cat((expansive_11_out, contracting_41_out), dim=1)) # [-1, 1024, 32, 32] -> [-1, 512, 32, 32]
		expansive_21_out = self.expansive_21(expansive_12_out) # [-1, 256, 64, 64]
		expansive_22_out = self.expansive_22(torch.cat((expansive_21_out, contracting_31_out), dim=1)) # [-1, 512, 64, 64] -> [-1, 256, 64, 64]
		expansive_31_out = self.expansive_31(expansive_22_out) # [-1, 128, 128, 128]
		expansive_32_out = self.expansive_32(torch.cat((expansive_31_out, contracting_21_out), dim=1)) # [-1, 256, 128, 128] -> [-1, 128, 128, 128]
		expansive_41_out = self.expansive_41(expansive_32_out)
		expansive_42_out = self.expansive_42(torch.cat((expansive_41_out, contracting_11_out), dim=1))
		x = expansive_42_out.view(expansive_42_out.size(0), -1)
		output_out = self.fc(x)
		return output_out

class BasicBlock(nn.Module):
	def __init__(self, in_channels, out_channels, downsample=None):
		
		super(BasicBlock,self).__init__()
		self.convfirst= nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1= nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2= nn.BatchNorm2d(out_channels)
		self.downsample = downsample
		
	def forward(self, x):
		
		identity = x
		out = self.convfirst(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		
		if self.downsample is not None:
			identity = self.downsample(x)
		out += identity #residual connection
		out = self.relu(out)
		return out
	
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(self, in_channels, out_channels, downsample = None):
        super().__init__()
        width = out_channels
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels * self.expansion, kernel_size=1, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class RESNET18(nn.Module):
		
	def __init__(self, num_classes):
		super(RESNET18, self).__init__()
		self.num_classes = num_classes
		
		self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride =2, padding =3, bias=False)
		self.bn1 = nn.BatchNorm2d(16)
		self.relu = nn.ReLU(inplace = True)
		self.maxpool = nn.MaxPool2d(kernel_size = 3, stride= 2, padding=1)
	
		self.layer11 = BasicBlock(16, 32, downsample=nn.Sequential(nn.Conv2d(16, 32, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(32)))
		self.layer12 = BasicBlock(32, 32)
		self.layer21 = BasicBlock(32, 64, downsample=nn.Sequential(nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(64)))
		self.layer22 = BasicBlock(64, 64)
		self.layer31 = BasicBlock(64, 128, downsample=nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(128)))
		self.layer32 = BasicBlock(128, 128) 
		self.layer41 = BasicBlock(128, 256, downsample=nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(256)))
		self.layer42 = BasicBlock(256, 256)

		self.avgpool = nn.AdaptiveAvgPool2d((1,1))
		self.fc = torch.nn.Linear(256, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity = 'relu')
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		
		x = self.layer11(x)
		x = self.layer12(x)
		x = self.layer21(x)
		x = self.layer22(x)
		x = self.layer31(x)
		x = self.layer32(x)
		x = self.layer41(x)
		x = self.layer42(x)
		
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.fc(x)
		return x

class RESNET18_small(nn.Module):
		
	def __init__(self, num_classes):
		super(RESNET18_small, self).__init__()
		self.num_classes = num_classes
		
		self.conv1 = nn.Conv2d(3, 8, kernel_size=7, stride =2, padding =3, bias=False)
		self.bn1 = nn.BatchNorm2d(8)
		self.relu = nn.ReLU(inplace = True)
		self.maxpool = nn.MaxPool2d(kernel_size = 3, stride= 2, padding=1)
	
		self.layer11 = BasicBlock(8, 16, downsample=nn.Sequential(nn.Conv2d(8, 16, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(16)))
		self.layer12 = BasicBlock(16, 16)
		self.layer21 = BasicBlock(16, 32, downsample=nn.Sequential(nn.Conv2d(16, 32, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(32)))
		self.layer22 = BasicBlock(32, 32)
		self.layer31 = BasicBlock(32, 64, downsample=nn.Sequential(nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(64)))
		self.layer32 = BasicBlock(64, 64)
		self.layer41 = BasicBlock(64, 128, downsample=nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(128)))
		self.layer42 = BasicBlock(128, 128)

		self.avgpool = nn.AdaptiveAvgPool2d((1,1))
		self.fc = torch.nn.Linear(128, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity = 'relu')
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		
		x = self.layer11(x)
		x = self.layer12(x)
		x = self.layer21(x)
		x = self.layer22(x)
		x = self.layer31(x)
		x = self.layer32(x)
		x = self.layer41(x)
		x = self.layer42(x)
		
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.fc(x)
		return x

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        # Image input_size=(3, 32, 32)
        self.layers = nn.Sequential(
            # input_size=(96, 55, 55)
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=2, padding=0), 
            nn.ReLU(), 
            # input_size=(96, 27, 27)
            nn.MaxPool2d(kernel_size=3, stride=2),
            # input_size=(256, 27, 27)
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=(5, 5), stride=1, padding=2),
            nn.ReLU(),
            # input_size=(256, 13, 13)
            nn.MaxPool2d(kernel_size=2, stride=2), 
            # input_size=(384, 13, 13)
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            # input_size=(384, 13, 13)
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            # input_size=(256, 13, 13)
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            # input_size=(256, 6, 6)
            nn.MaxPool2d(kernel_size=2, stride=2), 
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256*8*8, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
    
    def forward(self, x):
        x = self.layers(x)
        x = x.view(-1, 256*8*8)
        x = self.classifier(x)
        return x



class VGG16(nn.Module):
	
	def __init__(self, base_dim, num_classes=10):
		super(VGG16, self).__init__()
		self.feature = nn.Sequential(
			self.conv_2_block(3,base_dim), #64
			self.conv_2_block(base_dim,2*base_dim), #128
			self.conv_3_block(2*base_dim,4*base_dim), #256
			self.conv_3_block(4*base_dim,8*base_dim), #512
			self.conv_3_block(8*base_dim,8*base_dim), #512
		)
		# self.orth_basis = self.random_orthonormal_basis(4096, num_classes)
		self.connect_fc = nn.Linear(8*base_dim*1*1, num_classes)
		self.fc_relu = nn.ReLU(True)
		# self.connect_fc = nn.Sequential(
		# 	nn.Linear(8*base_dim*1*1, 10),
		# 	torch.transpose(self.orth_basis, 0, 1)
		# )
		# self.fc_layer = nn.Sequential(
        #     # nn.Linear(8*base_dim*1*1, 4096), # in case CIFAR10(with image size 32x32)
        #     # nn.Linear(8*base_dim*7*7, 4096), # in case IMAGENET(with image size 224x224)
		# 	self.connect_fc,
        #     nn.ReLU(True),
        #     # nn.Dropout(),
        #     # nn.Linear(4096, 4096),
        #     # nn.ReLU(True),
        #     # nn.Dropout(),
        #     # nn.Linear(4096, num_classes),
		# 	self.orth_basis
        # )
	
	def conv_2_block(self, in_dim,out_dim):
		model = nn.Sequential(
			nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1),
			nn.BatchNorm2d(out_dim),
			nn.ReLU(),
			nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
			nn.BatchNorm2d(out_dim),
			nn.ReLU(),
			nn.MaxPool2d(2,2)
		)
		return model
		
	def conv_3_block(self, in_dim,out_dim):
		model = nn.Sequential(
			nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1),
			nn.BatchNorm2d(out_dim),
        	nn.ReLU(),
        	nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
			nn.BatchNorm2d(out_dim),
        	nn.ReLU(),
        	nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
			nn.BatchNorm2d(out_dim),
        	nn.ReLU(),
        	nn.MaxPool2d(2,2)
    	)
		return model
	
	def random_orthonormal_basis(self, m, n):
		while True:
			A = np.random.randn(m, n)
			if np.linalg.matrix_rank(A) == n:
				Q, _ = np.linalg.qr(A)
				result = torch.FloatTensor(Q)
				result.requires_grad=False
				result = result.to('cuda')
				return result
	
	def forward(self, x):
		x = self.feature(x)
		x = x.view(x.size(0), -1)
		# x = self.fc_layer(x)
		x = self.connect_fc(x)
		# x = torch.matmul(x, torch.transpose(self.orth_basis, 0, 1))
		# x = self.fc_relu(x)
		# x = torch.matmul(x, self.orth_basis)
		return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, 
                 emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential( # 이미지를 패치 토큰으로 변환
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'))
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size)) # 0번째 토큰, 학습 가능
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size)) # 위치 임베딩
        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions

        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
	def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0., basis: Union[Tensor, None] = None):
		if basis is None:
			super().__init__(
				nn.Linear(emb_size, expansion * emb_size),
				nn.GELU(),
				nn.Dropout(drop_p),
				nn.Linear(expansion * emb_size, emb_size),
			)
		else:
			super().__init__(
				LinearWithBasis(torch.transpose(basis, 0, 1)),
				nn.GELU(),
				nn.Dropout(drop_p),
				LinearWithBasis(basis),
			)

class LinearWithBasis(nn.Module):
	def __init__(self, basis : Tensor): # basis means A^T
		super().__init__()
		self.basis = basis
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		factory_kwargs = {"device": device}
		self.in_features = basis.shape[0]
		self.out_features = basis.shape[1]
		dim = min(self.in_features, self.out_features)
		self.coff = torch.nn.parameter.Parameter(torch.empty((dim, dim), **factory_kwargs))
		self.reset_parameters()

	def reset_parameters(self) -> None:
		init.kaiming_uniform_(self.coff, a=math.sqrt(5))

	def forward(self, x : Tensor) -> Tensor:
		if self.in_features <= self.out_features:
			weight = self.coff @ self.basis
		else:
			weight = self.basis @ self.coff
		output = x @ weight
		return output


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
				 basis: Union[Tensor, None] = None,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p, basis=basis),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(nn.Sequential):
	def __init__(self, depth: int = 12, **kwargs):
		# self.basis = self.generate_orthonormal_basis(768, 192)
		self.basis = None
		super().__init__(*[TransformerEncoderBlock(basis=self.basis, **kwargs) for _ in range(depth)])
	
	# def generate_orthonormal_basis(self, n: int, p: int) -> Tensor:
	# 	A = np.random.randn(n, p)
	# 	Q, _ = np.linalg.qr(A)
	# 	basis = Tensor(Q[:, :p])
	# 	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# 	basis = basis.to(device)
	# 	basis.requires_grad = False
	# 	return basis

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes))

class ViT(nn.Sequential):
    def __init__(self,     
                in_channels: int = 3,
                patch_size: int = 4,
                emb_size: int = 768,
                img_size: int = 32,
                depth: int = 12,
                n_classes: int = 10,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )

class LeNet(nn.Module):
	def __init__(self, num_classes=10):
		super(LeNet, self).__init__()
		
		# in : (3,32,32) out : (6,28,28)
		self.cnn1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0) 
		self.relu1 = nn.ReLU()
		self.maxpool1 = nn.MaxPool2d(kernel_size=2) # 28/2 -> (6,14,14)
		
		# in : (6,14,14) out : (16,10,10)
		self.cnn2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0) 
		self.relu2 = nn.ReLU()
		self.maxpool2 = nn.MaxPool2d(kernel_size=2) # 10/2 -> (16,5,5)   
		
		self.fc1 = nn.Linear(16*5*5, 192) 
		self.relu5 = nn.ReLU()         
		self.fc2 = nn.Linear(192, 84)
		self.relu6 = nn.ReLU()
		self.fc3 = nn.Linear(84, num_classes)
		
	def forward(self, x):
		out = self.cnn1(x) 
		out = self.relu1(out)
		out = self.maxpool1(out)
		out = self.cnn2(out) 
		out = self.relu2(out) 
		out = self.maxpool2(out) 
		out = out.view(out.size(0), -1) # 완결연결층에 데이터를 전달하기 위해 데이터 형태를 1차원으로 바꿉니다.
		out = self.fc1(out)
		out = self.relu5(out)
		out = self.fc2(out)
		out = self.relu6(out)
		out = self.fc3(out)       
		return out
	
class LeNetRevised(nn.Module):
	def __init__(self, num_classes=10):
		super(LeNetRevised, self).__init__()
		
		# in : (3,32,32) out : (16,32,32)
		self.cnn1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1) 
		self.relu1 = nn.ReLU()
		self.maxpool1 = nn.MaxPool2d(kernel_size=2) # 32/2 -> (16,16,16)
		
		# in : (16,16,16) out : (16,16,16)
		self.cnn2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1) 
		self.relu2 = nn.ReLU()
		self.maxpool2 = nn.MaxPool2d(kernel_size=2) # 16/2 -> (16,8,8)

		# in : (16,8,8) out : (16,6,6)
		self.cnn3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=0) 
		self.relu3 = nn.ReLU()
		self.maxpool3 = nn.MaxPool2d(kernel_size=2) # 6/2 -> (16,3,3)
		
		self.fc1 = nn.Linear(16*3*3, num_classes) 
		
	def forward(self, x):
		out = self.cnn1(x) 
		out = self.relu1(out)
		out = self.maxpool1(out)
		out = self.cnn2(out) 
		out = self.relu2(out) 
		out = self.maxpool2(out)
		out = self.cnn3(out) 
		out = self.relu3(out) 
		out = self.maxpool3(out) 
		out = out.view(out.size(0), -1) # 완결연결층에 데이터를 전달하기 위해 데이터 형태를 1차원으로 바꿉니다.
		out = self.fc1(out)
		return out
	
class Per_patch_Fully_connected(nn.Module) :
    def __init__(self, input_size, patch_size, C) :
        super(Per_patch_Fully_connected, self).__init__()

        self.S = int((input_size[-2] * input_size[-1]) / (patch_size ** 2))
        self.x_dim_1_val = input_size[-3] * patch_size * patch_size
        self.projection_layer = nn.Linear(input_size[-3] * patch_size * patch_size,  C)

    def forward(self, x) :
        x = torch.reshape(x, (-1, self.S, self.x_dim_1_val)) 
        return self.projection_layer(x)
	
class token_mixing_MLP(nn.Module) : 
    def __init__(self, input_size) : 
        super(token_mixing_MLP, self).__init__()

        self.Layer_Norm = nn.LayerNorm(input_size[-2]) # C개의 값(columns)에 대해 각각 normalize 수행하므로 normalize되는 벡터의 크기는 S다. 
        self.MLP = nn.Sequential(
            nn.Linear(input_size[-2], input_size[-2]),
            nn.GELU(),
            nn.Linear(input_size[-2], input_size[-2])
        )

    def forward(self, x) :
        # layer_norm + transpose
        
        # [S x C]에서 column들을 가지고 연산하니까 Pytorch의 Layer norm을 적용하려면 transpose 하고 적용해야함. 
        output = self.Layer_Norm(x.transpose(2,1)) # transpose 후 Layer norm -> [C x S] 크기의 벡터가 나옴
        output = self.MLP(output)

        # [Batch x S x C] 형태로 transpose + skip connection
        output = output.transpose(2,1)

        return output + x
	
class channel_mixing_MLP(nn.Module) :
    def __init__(self, input_size) : # 
        super(channel_mixing_MLP, self).__init__()

        self.Layer_Norm = nn.LayerNorm(input_size[-1]) # S개의 벡터를 가지고 각각 normalize하니까 normalize되는 벡터의 크기는 C다

        self.MLP = nn.Sequential(
            nn.Linear(input_size[-1], input_size[-1]),
            nn.GELU(),
            nn.Linear(input_size[-1], input_size[-1])
        )
    
    def forward(self, x) :
        output = self.Layer_Norm(x)
        output = self.MLP(output)

        return output + x
	
class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True, **kwargs) -> None:
        super(Conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs) # kernel size = ...
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        if not self.activation:
            return self.batchnorm(self.conv(x))
        return self.relu(self.batchnorm(self.conv(x)))


class Res_block(nn.Module):
    def __init__(self, in_channels, red_channels, out_channels, is_plain=False):
        super(Res_block,self).__init__()
        self.relu = nn.ReLU()
        self.is_plain = is_plain
        
        if in_channels==64:
            self.convseq = nn.Sequential(
                                    Conv_block(in_channels, red_channels, kernel_size=1, padding=0),
                                    Conv_block(red_channels, red_channels, kernel_size=3, padding=1),
                                    Conv_block(red_channels, out_channels, activation=False, kernel_size=1, padding=0)
            )
            self.iden = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        elif in_channels == out_channels:
            self.convseq = nn.Sequential(
                                    Conv_block(in_channels, red_channels, kernel_size=1, padding=0),
                                    Conv_block(red_channels, red_channels, kernel_size=3, padding=1),
                                    Conv_block(red_channels, out_channels, activation=False, kernel_size=1, padding=0)
            )
            self.iden = nn.Identity()
        else:
            self.convseq = nn.Sequential(
                                    Conv_block(in_channels, red_channels, kernel_size=1, padding=0, stride=2),
                                    Conv_block(red_channels, red_channels, kernel_size=3, padding=1),
                                    Conv_block(red_channels, out_channels, activation=False, kernel_size=1, padding=0)
                
            )
            self.iden = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        
    def forward(self, x):
        y = self.convseq(x)
        if self.is_plain:
            x = y
        else:
            x = y + self.iden(x)
        x = self.relu(x)  # relu(skip connection)
        return x
	

class RESNET50(nn.Module):
    def __init__(self, in_channels=3 , num_classes=1000, is_plain=False):
        self.num_classes = num_classes
        super(RESNET50, self).__init__()
        self.conv1 = Conv_block(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv2_x = nn.Sequential(
                                        Res_block(64, 64, 256, is_plain),
                                        Res_block(256, 64, 256, is_plain),
                                        Res_block(256, 64, 256, is_plain)
        )
        
        self.conv3_x = nn.Sequential(
                                        Res_block(256, 128, 512, is_plain),
                                        Res_block(512, 128, 512, is_plain),
                                        Res_block(512, 128, 512, is_plain),
                                        Res_block(512, 128, 512, is_plain)
        )

        self.conv4_x = nn.Sequential(
                                        Res_block(512, 256, 1024, is_plain),
                                        Res_block(1024, 256, 1024, is_plain),
                                        Res_block(1024, 256, 1024, is_plain),
                                        Res_block(1024, 256, 1024, is_plain),
                                        Res_block(1024, 256, 1024, is_plain),
                                        Res_block(1024, 256, 1024, is_plain)
        )
        
        self.conv5_x = nn.Sequential(
                                        Res_block(1024, 512, 2048, is_plain),
                                        Res_block(2048, 512, 2048, is_plain),
                                        Res_block(2048, 512, 2048, is_plain),
        )

        self.avgpool = nn.AvgPool2d((1,1))
        self.fc = nn.Linear(2048,num_classes)

    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x