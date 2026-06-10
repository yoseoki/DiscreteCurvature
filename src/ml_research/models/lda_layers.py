import torch
import torch.nn as nn
import numpy as np

'''
LDAConv2d : 「층 별로」 부분공간 기저를 로드하여 저차원 학습을 진행할 때 사용하는 합성곱 층
LDA2Conv2d : 「모델 전체에 대해」 부분공간 기저를 로드하여 저차원 학습을 진행할 때 사용하는 합성곱 층
LDAFC : 「층 별로」 부분공간 기저를 로드하여 저차원 학습을 진행할 때 사용하는 전결합 층
LDA2FC : 「모델 전체에 대해」 부분공간 기저를 로드하여 저차원 학습을 진행할 때 사용하는 전결합 층

「층 별로」 : import_basis 라는 메소드를 통해 기저 행렬을 로드, initFlag 라는 인수를 통해 「제안 초기화 수법」을
            적용할 것인지 선택가능.
「모델 전체에 대해」 : set_weight 라는 메소드를 통해 웨이트와 편향을 로드, 당연히 학습되어지는 대상은 아님.
'''

class LDAConv2d(nn.Conv2d):
	def __init__(self, apprSize, inputCh, outputCh, kernelSize, paddingSize):
		super().__init__(inputCh, outputCh, kernelSize, padding=paddingSize)
		self.weight.requires_grad = False
		self.bias.requires_grad = False

		self.weightDim = inputCh * outputCh * kernelSize * kernelSize
		self.biasDim = outputCh
		self.projDim = self.weightDim + self.biasDim
		self.apprDim = apprSize

		self.proj = torch.randn(self.projDim, apprSize, requires_grad=False)
		norm = torch.linalg.norm(self.proj, dim=0)
		self.proj = torch.div(self.proj, norm)

		self.apprWeight = torch.nn.Parameter(torch.randn(apprSize, 1), requires_grad=True)
		self._parameters["apprWeight"] = self.apprWeight
	
	def forward(self, input):
		tmp = torch.matmul(self.proj, self.apprWeight)
		tmp_weight = tmp[:self.weightDim,:]
		tmp_bias = tmp[self.weightDim:,:]
		tmp_bias = torch.squeeze(tmp_bias)
		tmp_weight = tmp_weight.view([self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[0]])
		return self._conv_forward(input, self.weight + tmp_weight, self.bias + tmp_bias)
	
	def import_basis(self, basis_matrix, initFlag):

		if self.apprDim != basis_matrix.shape[1]:
			print("dimension of appoximate space is not equal to number of basis!")
			return
		
		self.proj = torch.FloatTensor(basis_matrix)
		self.proj.requires_grad = False
		
		if initFlag:
			tmp = np.linalg.inv(np.transpose(basis_matrix) @ basis_matrix) @ np.transpose(basis_matrix)
			init_vec = np.sqrt(2 / basis_matrix.shape[0]) * np.random.randn(basis_matrix.shape[0], 1)
			init_vec_appr  = tmp @ init_vec
			self.apprWeight = torch.nn.Parameter(torch.FloatTensor(init_vec_appr), requires_grad=True)
			
class LDA2Conv2d(nn.Conv2d):
	def __init__(self, inputCh, outputCh, kernelSize, paddingSize):
		super().__init__(inputCh, outputCh, kernelSize, padding=paddingSize)
		self.weight.requires_grad = False
		self.bias.requires_grad = False

		self.tmpWeight = None
		self.tmpBias = None

	def set_weight(self, weight, bias):
		self.tmpWeight = weight.view([self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[0]])
		self.tmpBias = torch.squeeze(bias)
	
	def forward(self, input):
		return self._conv_forward(input, self.tmpWeight, self.tmpBias)
	
class LDAFC(nn.Module):
	def __init__(self, inputDim, outputDim, apprSize, bias=True):
		super().__init__()
		self.weight = torch.randn(outputDim, inputDim, requires_grad=False)
		self.bias = torch.randn(outputDim, 1, requires_grad=False)

		self.inputDim = inputDim
		self.outputDim = outputDim

		self.projDim = inputDim * outputDim + outputDim
		self.weightDim = inputDim * outputDim
		self.biasDim = outputDim
		self.apprDim = apprSize

		self.proj = torch.randn(self.projDim, apprSize, requires_grad=False)
		norm = torch.linalg.norm(self.proj, dim=0)
		self.proj = torch.div(self.proj, norm)
		self.apprWeight = torch.nn.Parameter(torch.randn(apprSize, 1), requires_grad=True)
		self._parameters["apprWeight"] = self.apprWeight
	
	def forward(self, input):
		tmp = torch.matmul(self.proj, self.apprWeight)
		tmp_weight = tmp[:self.weightDim,:]
		tmp_bias = tmp[self.weightDim:,:]
		tmp_weight = tmp_weight.view([self.outputDim, self.inputDim])
		result = torch.matmul(input, torch.transpose(tmp_weight, 0, 1)) + torch.transpose(tmp_bias, 0, 1)
		return result
	
	def import_basis(self, basis_matrix, initFlag):

		if self.apprDim != basis_matrix.shape[1]:
			print("dimension of appoximate space is not equal to number of basis!")
			return
		
		self.proj = torch.FloatTensor(basis_matrix)
		self.proj.requires_grad = False
		
		if initFlag:
			tmp = np.linalg.inv(np.transpose(basis_matrix) @ basis_matrix) @ np.transpose(basis_matrix)
			init_vec = np.sqrt(2 / basis_matrix.shape[0]) * np.random.randn(basis_matrix.shape[0], 1)
			init_vec_appr  = tmp @ init_vec
			
			self.apprWeight = torch.nn.Parameter(torch.FloatTensor(init_vec_appr), requires_grad=True)
		
class LDA2FC(nn.Module):
	def __init__(self, inputDim, outputDim):
		super().__init__()
		self.tmpWeight = torch.randn(outputDim, inputDim, requires_grad=False)
		self.tmpBias = torch.randn(outputDim, 1, requires_grad=False)

		self.inputDim = inputDim
		self.outputDim = outputDim

	def set_weight(self, weight, bias):
		self.tmpWeight = weight.view([self.outputDim, self.inputDim])
		self.tmpBias = bias
	
	def forward(self, input):
		result = torch.matmul(self.tmpWeight, torch.transpose(input, 0, 1)) + self.tmpBias
		return torch.transpose(result, 0, 1)