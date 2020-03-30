import torch;
import torch.nn as nn;
import torch.nn.functional as F;
import numpy as np;

torch.set_printoptions(precision=10);

epsilon = 0.0001;
learningRate = 0.002;
yHats = [];

inputData = torch.Tensor([[1, 0], [0.5, 0.25], [0, 1]]);

V = torch.Tensor([[-2, 1], [-1, 0]]);
#V = torch.Tensor([[2, -1], [1, 0]]);

W = torch.Tensor([[1, -1], [0, 2]]);

U = torch.Tensor([[-1, 0], [1, -2]]);

c = torch.Tensor([0.5, -0.5]);

b = torch.Tensor([-1, 1]);
#b = torch.Tensor([1, -1]);

h = [];


class LSTM(nn.Module):
	def __init__(self, inputSize, hiddenSize, outputSize):
		super().__init__();
		
		self.hiddenSize = hiddenSize;
		self.i2h = nn.Linear(inputSize + hiddenSize, hiddenSize);
		self.h2o = nn.Linear(hiddenSize, outputSize);
		self.tanh = nn.Tanh();
		self.softmax = nn.Softmax(dim=1);
		self.sigmoid = nn.Sigmoid();


	def forward(self, input, cell0):
		#print("input: {}".format(input.shape));
		#print("cell: {}".format(cell0.shape));
		combined = torch.cat((cell0, input), dim=1);
		#print("combined: {}".format(combined));
		cell0 = self.i2h(combined);
		f = self.sigmoid(cell0);
		i = self.sigmoid(cell0);
		o = self.sigmoid(cell0);
		#print(cell0);
		g = self.sigmoid(cell0);
		cell = f*cell0 + i*g;
		h.append(cell.detach());
		cell = o*self.tanh(cell);
		#print("hidden 2nd: {}".format(hidden));
		x = self.h2o(cell);
		output = self.softmax(x);
		#print("output: {}".format(output.shape));

		return output, cell;

	def initCell(self):
		return torch.zeros(1, self.hiddenSize);


def lossFunction(yHat):
	yHat = yHat.squeeze();
	return ((yHat[0]-0.5)*(yHat[0]-0.5)) - torch.log(yHat[1]);


def initParamsHidden(m):
	weights = torch.cat((W, U), dim=1);

	if type(m) == nn.Linear:
		m.weight.data = weights;
		m.bias.data = b.clone();


def initParamsOutput(m):	
	if type(m) == nn.Linear:
		m.weight.data = V.clone();
		m.bias.data = c.clone();


def reinitBiasValues(addition, whichVals):
	def reinitBiasValue(m):
		if type(m) == nn.Linear:
			if whichVals == 2:
				m.bias.data[0] = m.bias[0]+addition[0];
				m.bias.data[1] = m.bias[1]+addition[1];
			else:
				m.bias.data[whichVals] = m.bias[whichVals]+addition[whichVals];

	return reinitBiasValue;


def train(model, trainData):
	cell = model.initCell();

	for x in range(len(trainData)):
		(output, cell) = model(trainData[x], cell);
		print("y^({}) is {}".format(x+1, str(output.detach())));
		print();
		yHats.append(output.detach());
	
	loss = lossFunction(output);

	return loss;	



def calculateGradients(nInp, nHid, nOut):
	model1 = LSTM(nInp, nHid, nOut);
	model1.i2h.apply(initParamsHidden);
	model1.h2o.apply(initParamsOutput);

	model2 = LSTM(nInp, nHid, nOut);
	model2.i2h.apply(initParamsHidden);
	model2.h2o.apply(initParamsOutput);

	model3 = LSTM(nInp, nHid, nOut);
	model3.i2h.apply(initParamsHidden);
	model3.h2o.apply(initParamsOutput);

	model4 = LSTM(nInp, nHid, nOut);
	model4.i2h.apply(initParamsHidden);
	model4.h2o.apply(initParamsOutput);

	model1.i2h.apply(reinitBiasValues((-epsilon, None), 0));
	model2.i2h.apply(reinitBiasValues((None, -epsilon), 1));
	model3.i2h.apply(reinitBiasValues((epsilon, None), 0));
	model4.i2h.apply(reinitBiasValues((None, epsilon), 1));

	#print("{0:0.8f}".format(model1.i2h.bias[0]));
	#print("{0:0.8f}".format(model1.i2h.bias[1]));
	#print("{0:0.8f}".format(model2.i2h.bias[0]));
	#print("{0:0.8f}".format(model2.i2h.bias[1]));
	#print("{0:0.8f}".format(model3.i2h.bias[0]));
	#print("{0:0.8f}".format(model3.i2h.bias[1]));
	#print("{0:0.8f}".format(model4.i2h.bias[0]));
	#print("{0:0.8f}".format(model4.i2h.bias[1]));


	lossBias1Sub = train(model1, inputData);
	lossBias2Sub = train(model2, inputData);
	lossBias1Add = train(model3, inputData);
	lossBias2Add = train(model4, inputData);

	#print("{0:0.10f}".format(lossBias1Add));
	#print("{0:0.10f}".format(lossBias1Sub));

	#print("{0:0.10f}".format(lossBias2Add));
	#print("{0:0.10f}".format(lossBias2Sub));

	derivativeWrtB1 = (lossBias1Add - lossBias1Sub)/(2*epsilon);
	derivativeWrtB2 = (lossBias2Add - lossBias2Sub)/(2*epsilon);

	return (derivativeWrtB1, derivativeWrtB2);



def gradientDescent(derivLossWrtB1, derivLossWrtB2):
	subFromB1 = derivLossWrtB1 * b[0] * learningRate;
	subFromB2 = derivLossWrtB2 * b[1] * learningRate;
	
	return (subFromB1, subFromB2);


if __name__ == '__main__':

	inputData.unsqueeze_(1);
	
	nInp, nHid, nOut = 2, 2, 2;
	model = LSTM(nInp, nHid, nOut);
	model.i2h.apply(initParamsHidden);
	model.h2o.apply(initParamsOutput);
	

	loss = train(model, inputData);	
	#print("loss is {}".format(loss));

	(derivLossWrtB1, derivLossWrtB2) = calculateGradients(nInp, nHid, nOut);
	#print();
	#print("Derivative of Loss wrt B1: {}".format(derivLossWrtB1));
	#print();
	#print("Derivative of Loss wrt B2: {}".format(derivLossWrtB2));
	#print();


	(subFromB1, subFromB2) = gradientDescent(derivLossWrtB1, derivLossWrtB2);
	
	model.i2h.apply(reinitBiasValues((-subFromB1, subFromB2), 2));

	lossAfterDescent = train(model, inputData);
	
	print("new loss is {}".format(lossAfterDescent));
	print();

	print("Difference in the loss value: {}".format(loss - lossAfterDescent));
	print();
