import torch
import torch.nn as nn
import numpy as np
from ml_research.models import lda_layers as lm

class FC_MNIST(nn.Module):

	def __init__(self, itemNum):
		super(FC_MNIST, self).__init__()
		
		self.layer1 = torch.nn.Sequential(
			nn.Linear(28 * 28, 14 * 14, bias=True),
			nn.ReLU())
		self.layer2 = torch.nn.Sequential(
			nn.Linear(14 * 14, 36, bias=True),
			nn.ReLU())
		self.fc = nn.Linear(36, itemNum, bias=True)

	def forward(self, x):
		x = x.view(x.size(0), -1)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.fc(x)
		return x
	
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

class FC_MNIST_LDA2(nn.Module):
	def __init__(self, itemNum, apprNum):
		super(FC_MNIST_LDA2, self).__init__()
		self.inputDim = 784
		self.mediumDim1 = 196
		self.mediumDim2 = 36
		self.outputDim = itemNum
		self.projDim = self.inputDim*self.mediumDim1 + self.mediumDim1 + self.mediumDim1*self.mediumDim2 + self.mediumDim2 + self.mediumDim2*self.outputDim + self.outputDim
		self.apprDim = apprNum
		self.fc1 = lm.LDA2FC(self.inputDim, self.mediumDim1)
		self.fc2 = lm.LDA2FC(self.mediumDim1, self.mediumDim2)
		self.fc3 = lm.LDA2FC(self.mediumDim2, self.outputDim)
		self.proj = torch.randn(self.projDim, apprNum, requires_grad=False)
		self.apprWeight = torch.nn.Parameter(torch.randn(apprNum, 1), requires_grad=True)
		self._parameters["apprWeight"] = self.apprWeight
	
	def forward(self, x):
		x = x.view(x.size(0), -1)
		
		tmp = torch.matmul(self.proj, self.apprWeight)
		tmpIdx1 = self.inputDim*self.mediumDim1
		tmpIdx2 = tmpIdx1 + self.mediumDim1
		self.fc1.set_weight(tmp[:tmpIdx1,:], tmp[tmpIdx1:tmpIdx2,:])
		
		tmpIdx3 = tmpIdx2 + self.mediumDim1*self.mediumDim2
		tmpIdx4 = tmpIdx3 + self.mediumDim2
		self.fc2.set_weight(tmp[tmpIdx2:tmpIdx3,:], tmp[tmpIdx3:tmpIdx4,:])
		
		tmpIdx5 = tmpIdx4 + self.mediumDim2*self.outputDim
		tmpIdx6 = tmpIdx5 + self.outputDim
		self.fc3.set_weight(tmp[tmpIdx4:tmpIdx5,:], tmp[tmpIdx5:tmpIdx6,:])

		x = self.fc1(x)
		x = self.fc2(x)
		x = self.fc3(x)
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