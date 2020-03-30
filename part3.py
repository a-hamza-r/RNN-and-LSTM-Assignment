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


class RNN(nn.Module):
	def __init__(self, inputSize, hiddenSize, outputSize):
		super().__init__();
		
		self.hiddenSize = hiddenSize;
		self.i2h = nn.Linear(inputSize + hiddenSize, hiddenSize);
		self.h2o = nn.Linear(hiddenSize, outputSize);
		self.tanh = nn.Tanh();
		self.softmax = nn.Softmax(dim=1);


	def forward(self, input, hidden0):
		#print("input: {}".format(input.shape));
		#print("hidden: {}".format(hidden0.shape));
		combined = torch.cat((hidden0, input), dim=1);
		#print("combined: {}".format(combined));
		hidden0 = self.i2h(combined);
		#print(hidden0);
		hidden = self.tanh(hidden0);
		h.append(hidden.detach());
		#print("hidden 2nd: {}".format(hidden));
		x = self.h2o(hidden);
		output = self.softmax(x);
		#print("output: {}".format(output.shape));

		return output, hidden;

	def initHidden(self):
		return torch.zeros(1, self.hiddenSize);


def lossFunction(yHat):
	yHat = yHat.squeeze();
	return ((yHat[0]-0.5)*(yHat[0]-0.5)) - torch.log(yHat[1]);


def lossFunctionManual(yHat):
	yHat = yHat.squeeze();
	return -torch.log(yHat[1]);


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

'''
def reinitBias1HiddenSub(m):
	if type(m) == nn.Linear:
		m.bias.data[0] = m.bias.data[0]-epsilon;


def reinitBias2HiddenSub(m):
	if type(m) == nn.Linear:
		m.bias.data[1] = m.bias.data[1]-epsilon;


def reinitBias1HiddenAdd(m):
	if type(m) == nn.Linear:
		m.bias.data[0] = m.bias.data[0]+epsilon;


def reinitBias2HiddenAdd(m):
	if type(m) == nn.Linear:
		m.bias.data[1] = m.bias.data[1]+epsilon;
'''

def train(model, trainData):
	hidden = model.initHidden();

	for x in range(len(trainData)):
		(output, hidden) = model(trainData[x], hidden);
		#print("y^({}) is {}".format(x+1, str(output.detach())));
		yHats.append(output.detach());
	
	loss = lossFunction(output);

	return loss;
	

def centralDifferenceMethod(nInp, nHid, nOut):
	model1 = RNN(nInp, nHid, nOut);
	model1.i2h.apply(initParamsHidden);
	model1.h2o.apply(initParamsOutput);

	model2 = RNN(nInp, nHid, nOut);
	model2.i2h.apply(initParamsHidden);
	model2.h2o.apply(initParamsOutput);

	model3 = RNN(nInp, nHid, nOut);
	model3.i2h.apply(initParamsHidden);
	model3.h2o.apply(initParamsOutput);

	model4 = RNN(nInp, nHid, nOut);
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

	approxDerivativeWrtB1 = (lossBias1Add - lossBias1Sub)/(2*epsilon);
	approxDerivativeWrtB2 = (lossBias2Add - lossBias2Sub)/(2*epsilon);
	print();
	print("Approximate derivative of Loss wrt B1: {}".format(approxDerivativeWrtB1));
	print();
	print("Approximate derivative of Loss wrt B2: {}".format(approxDerivativeWrtB2));
	print();


def derivLossWrtO(time):
	derivLossWrtY1 = 2*(yHats[time][0] - 0.5);
	derivLossWrtY2 = -1/yHats[time][1];
	
	derivY1WrtO1 = yHats[time][0]*(1 - yHats[time][0]);
	derivY2WrtO2 = yHats[time][1]*(1 - yHats[time][1]);
	derivY1WrtO2 = -yHats[time][0]*yHats[time][1];
	derivY2WrtO1 = -yHats[time][0]*yHats[time][1];

	derivLossWrtO1 = derivLossWrtY1 * derivY1WrtO1 + derivLossWrtY2 * derivY2WrtO1;
	derivLossWrtO2 = derivLossWrtY1 * derivY1WrtO2 + derivLossWrtY2 * derivY2WrtO2;

	return (derivLossWrtO1, derivLossWrtO2);
	#return (yHats[2][0], yHats[2][1]-1);



def findDerivativeLossWrtB1():
		 	
	derivA1_1stWrtB1 = 1; 
	derivA2_1stWrtB1 = 0;

	derivH1_1stWrtB1 = (1-h[0][0]*h[0][0])*derivA1_1stWrtB1;
	derivH2_1stWrtB1 = (1-h[0][1]*h[0][1])*derivA2_1stWrtB1;

	print("Derivative of a1^(1) wrt b1: {}".format(derivA1_1stWrtB1));	
	print("Derivative of a2^(1) wrt b1: {}".format(derivA2_1stWrtB1));

	print("Derivative of h1^(1) wrt b1: {}".format(derivH1_1stWrtB1));
	print("Derivative of h2^(1) wrt b1: {}".format(derivH2_1stWrtB1));

	derivA1_2ndWrtB1 = 1 + (W[0][0]*derivH1_1stWrtB1) + (W[0][1]*derivH2_1stWrtB1);
	derivA2_2ndWrtB1 = (W[1][0]*derivH1_1stWrtB1) + (W[1][1]*derivH2_1stWrtB1);

	derivH1_2ndWrtB1 = (1-h[1][0]*h[1][0])*derivA1_2ndWrtB1;
	derivH2_2ndWrtB1 = (1-h[1][1]*h[1][1])*derivA2_2ndWrtB1;

	print("Derivative of a1^(2) wrt b1: {}".format(derivA1_2ndWrtB1));	
	print("Derivative of a2^(2) wrt b1: {}".format(derivA2_2ndWrtB1));
	
	print("Derivative of h1^(2) wrt b1: {}".format(derivH1_2ndWrtB1));
	print("Derivative of h2^(2) wrt b1: {}".format(derivH2_2ndWrtB1));

	derivA1_3rdWrtB1 = 1+W[0][0]*derivH1_2ndWrtB1+W[0][1]*derivH2_2ndWrtB1;
	derivA2_3rdWrtB1 = W[1][0]*derivH1_2ndWrtB1+W[1][1]*derivH2_2ndWrtB1;
	
	derivH1_3rdWrtB1 = (1-h[2][0]*h[2][0])*derivA1_3rdWrtB1;
	derivH2_3rdWrtB1 = (1-h[2][1]*h[2][0])*derivA2_3rdWrtB1;

	print("Derivative of a1^(3) wrt b1: {}".format(derivA1_3rdWrtB1));	
	print("Derivative of a2^(3) wrt b1: {}".format(derivA2_3rdWrtB1));

	print("Derivative of h1^(3) wrt b1: {}".format(derivH1_3rdWrtB1));
	print("Derivative of h2^(3) wrt b1: {}".format(derivH2_3rdWrtB1));

	(derivLossWrtO1_3rd, derivLossWrtO2_3rd) = derivLossWrtO(2);

	derivLossWrtH1_3rd = derivLossWrtO1_3rd*V[0][0] + derivLossWrtO2_3rd*V[1][0];
	derivLossWrtH2_3rd = derivLossWrtO1_3rd*V[0][1] + derivLossWrtO2_3rd*V[1][1];
	
	print("Derivative of loss wrt h1^(3): {}".format(derivLossWrtH1_3rd));
	print("Derivative of loss wrt h2^(3): {}".format(derivLossWrtH2_3rd));

	derivLossWrtB1 = derivLossWrtH1_3rd*derivH1_3rdWrtB1 + derivLossWrtH2_3rd*derivH2_3rdWrtB1;
	print("Derivative of Loss wrt b1: {}".format(derivLossWrtB1));
	print();

	return derivLossWrtB1;


def findDerivativeLossWrtB2():
	
	derivA2_1stWrtB2 = 1; 
	derivA1_1stWrtB2 = 0;
		
	derivH1_1stWrtB2 = (1-h[0][0]*h[0][0])*derivA1_1stWrtB2;
	derivH2_1stWrtB2 = (1-h[0][1]*h[0][1])*derivA2_1stWrtB2;
	
	print("Derivative of a1^(1) wrt b2: {}".format(derivA1_1stWrtB2));	
	print("Derivative of a2^(1) wrt b2: {}".format(derivA2_1stWrtB2));

	print("Derivative of h1^(1) wrt b2: {}".format(derivH1_1stWrtB2));
	print("Derivative of h2^(1) wrt b2: {}".format(derivH2_1stWrtB2));

	derivA2_2ndWrtB2 = 1 + (W[1][0]*derivH1_1stWrtB2) + (W[1][1]*derivH2_1stWrtB2);
	derivA1_2ndWrtB2 = (W[0][0]*derivH1_1stWrtB2) + (W[0][1]*derivH2_1stWrtB2);

	derivH1_2ndWrtB2 = (1-h[1][0]*h[1][0])*derivA1_2ndWrtB2;
	derivH2_2ndWrtB2 = (1-h[1][1]*h[1][1])*derivA2_2ndWrtB2;

	print("Derivative of a1^(2) wrt b2: {}".format(derivA1_2ndWrtB2));	
	print("Derivative of a2^(2) wrt b2: {}".format(derivA2_2ndWrtB2));

	print("Derivative of h1^(2) wrt b2: {}".format(derivH1_2ndWrtB2));
	print("Derivative of h2^(2) wrt b2: {}".format(derivH2_2ndWrtB2));

	derivA1_3rdWrtB2 = W[0][0]*derivH1_2ndWrtB2+W[0][1]*derivH2_2ndWrtB2;
	derivA2_3rdWrtB2 = 1+W[1][0]*derivH1_2ndWrtB2+W[1][1]*derivH2_2ndWrtB2;

	derivH1_3rdWrtB2 = (1-h[2][0]*h[2][0])*derivA1_3rdWrtB2;
	derivH2_3rdWrtB2 = (1-h[2][1]*h[2][1])*derivA2_3rdWrtB2;

	print("Derivative of a1^(3) wrt b2: {}".format(derivA1_3rdWrtB2));	
	print("Derivative of a2^(3) wrt b2: {}".format(derivA2_3rdWrtB2));
	
	print("Derivative of h1^(3) wrt b2: {}".format(derivH1_3rdWrtB2));
	print("Derivative of h2^(3) wrt b2: {}".format(derivH2_3rdWrtB2));

	(derivLossWrtO1_3rd, derivLossWrtO2_3rd) = derivLossWrtO(2);
	
	derivLossWrtH1_3rd = derivLossWrtO1_3rd*V[0][0] + derivLossWrtO2_3rd*V[1][0];
	derivLossWrtH2_3rd = derivLossWrtO1_3rd*V[0][1] + derivLossWrtO2_3rd*V[1][1];
	
	print("Derivative of loss wrt h1^(3): {}".format(derivLossWrtH1_3rd));
	print("Derivative of loss wrt h2^(3): {}".format(derivLossWrtH2_3rd));

	derivLossWrtB2 = derivLossWrtH1_3rd*derivH1_3rdWrtB2 + derivLossWrtH2_3rd*derivH2_3rdWrtB2;
	print("Derivative of Loss wrt b2: {}".format(derivLossWrtB2));
	print();

	return derivLossWrtB2;


def gradientDescent(derivLossWrtB1, derivLossWrtB2):
	subFromB1 = derivLossWrtB1 * b[0] * learningRate;
	subFromB2 = derivLossWrtB2 * b[1] * learningRate;
	
	return (subFromB1, subFromB2);


if __name__ == '__main__':

	inputData.unsqueeze_(1);
	
	nInp, nHid, nOut = 2, 2, 2;
	model = RNN(nInp, nHid, nOut);
	model.i2h.apply(initParamsHidden);
	model.h2o.apply(initParamsOutput);

	loss = train(model, inputData);	
	print("loss is {}".format(loss));
	
	centralDifferenceMethod(nInp, nHid, nOut);

	yHats = torch.stack(yHats).squeeze();
	h = torch.stack(h).squeeze();
	
	derivLossWrtB1 = findDerivativeLossWrtB1();
	derivLossWrtB2 = findDerivativeLossWrtB2();

	(subFromB1, subFromB2) = gradientDescent(derivLossWrtB1, derivLossWrtB2);

	model.i2h.apply(reinitBiasValues((-subFromB1, -subFromB2), 2));
	
	yHats = [];
	h = [];

	lossAfterDescent = train(model, inputData);	
	print("loss is {}".format(lossAfterDescent));
	print();

	print("Difference in the loss value: {}".format(loss - lossAfterDescent));
	print();
