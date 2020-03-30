import torch;
import torch.nn as nn;
import torch.nn.functional as F;
import numpy as np; 
from torchvision import datasets, transforms, models;
import matplotlib.pyplot as plt;

class Network(nn.Module):
	def __init__(self):
		super().__init__();

		self.conv1 = nn.Conv2d(1, 16, 3, 1);
		self.conv2 = nn.Conv2d(16, 32, 3, 1);
		self.conv3 = nn.Conv2d(32, 64, 3, 1);
		self.dropout1 = nn.Dropout2d(0.25);
		self.dropout2 = nn.Dropout2d(0.5);
		self.fc1 = nn.Linear(1600, 128);
		self.fc2 = nn.Linear(128, 10);
		self.pool = nn.MaxPool2d(2);
		
		self.relu = nn.ReLU();
		self.tanh = nn.Tanh();
		self.sigmoid = nn.Sigmoid();
		self.log_softmax = nn.LogSoftmax(dim=1);

	def forward(self, x):
		x = self.conv1(x);
		x = self.relu(x);
		x = self.conv2(x);
		x = self.pool(x);
		x = self.dropout1(x);
		x = self.conv3(x);
		x = self.pool(x);
		x = self.dropout2(x);
		x = torch.flatten(x, 1);
		x = self.fc1(x);
		x = self.relu(x);
		x = self.dropout2(x);
		x = self.fc2(x);
		x = self.log_softmax(x);

		return x;

def train(model, trainLoader, optimizer, epoch):
	for batchIdx, (data, target) in enumerate(trainLoader):
		optimizer.zero_grad();
		output = model(data);
		loss = F.nll_loss(output, target);
		loss.backward();
		optimizer.step();
		if batchIdx % 10 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			epoch, batchIdx * len(data), len(trainLoader.dataset),
			100. * batchIdx / len(trainLoader), loss.item()));


def test(model, testLoader):
	testLoss = 0;
	correct = 0;
	with torch.no_grad():
		for data, target in testLoader:
			output = model(data);
			testLoss += F.nll_loss(output, target, reduction='sum').item();
			pred = output.argmax(dim=1, keepdim=True);
			correct += pred.eq(target.view_as(pred)).sum().item();

	testLoss /= len(testLoader.dataset);
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
	testLoss, correct, len(testLoader.dataset),
	100. * correct / len(testLoader.dataset)));
			
def validate(model, trainLoader):
    testLoss = 0;
    correct = 0;
    with torch.no_grad():
        for batchIdx, (data, target) in enumerate(trainLoader):
        	output = model(data);
        	testLoss += F.nll_loss(output, target, reduction='sum').item();  # sum up batch loss
        	pred = output.argmax(dim=1, keepdim=True);  # get the index of the max log-probability
        	correct += pred.eq(target.view_as(pred)).sum().item();

    testLoss /= len(trainLoader.dataset);
    print('\nValidate set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        testLoss, correct, len(trainLoader.dataset),
        100. * correct / len(trainLoader.dataset)));

def plotWeights(layer):
	weights = layer.weight.data;
	numPlots = weights.shape[0];
	print(weights.shape)
	numColumns = 8;
	
	numRows = 1 + numPlots/numColumns;

	npImage = np.array(weights.numpy(), np.float32);
	for j in range(weights.shape[1]):
		figure = plt.figure(figsize=(numColumns, numRows));
	
		for i in range(weights.shape[0]):
			ax1 = figure.add_subplot(numRows, numColumns, i+1);
			npImage = np.array(weights[i][j].numpy());
			# npImage = (npImage - np.mean(npImage))/np.std(npImage);
			#npImage = np.minimum(1, np.maximum(0, (npImage + 0.5)));
			ax1.imshow(npImage, cmap="gray");
			ax1.set_title('Filter {}'.format(str(i)));
			ax1.axis('off');
			ax1.set_xticklabels([]);
			ax1.set_yticklabels([]);
		
		plt.tight_layout();
		plt.show();


def visualizeFeatureMap(activation):
	numColumns = 12;
	numPlots = activation.size(0);

	numRows = 1 + int(numPlots/numColumns);
	figure = plt.figure(figsize=(numColumns, numRows));

	for idx in range(numPlots):
		ax1 = figure.add_subplot(numRows, numColumns, idx + 1);
		ax1.imshow(activation[idx], cmap='gray');
		ax1.set_title('Channel {}'.format(idx));
		ax1.axis('off');
		ax1.set_xticklabels([]);
		ax1.set_yticklabels([]);

	plt.tight_layout();
	plt.show();


def findSpecificLabel(digit, model, testLoader):
	for _, (data, target) in enumerate(testLoader):
		output = model(data);
		pred = output.argmax(dim=1);
		allDigits = (pred == digit).nonzero().squeeze();
		accurateDigit = torch.tensor([x for x in allDigits if target[x] == digit], dtype=torch.int32)[0];
		return (accurateDigit, data);


def plotFeatureMaps(model, testLoader):
	activation = {};
	
	def getActivation(layerName):
		def hook(model, input, output):
			activation[layerName] = output.detach();
		return hook;
	
	(accurateZero, data) = findSpecificLabel(0, model, testLoader);
	(accurateEight, data1) = findSpecificLabel(8, model, testLoader);

	#model.conv2.register_forward_hook(getActivation('conv2'));
	model.conv3.register_forward_hook(getActivation('conv3'));
	plt.imshow(data[accurateZero].squeeze(), cmap="gray");
	plt.show();
	plt.imshow(data1[accurateEight].squeeze(), cmap="gray");
	plt.show();
	sampleZero = data[accurateZero].unsqueeze(0);
	sampleEight = data1[accurateEight].unsqueeze(0);
	
	outputZero = model(sampleZero);
	#act = activation['conv2'].squeeze();
	#visualizeFeatureMap(act);
	act = activation['conv3'].squeeze();
	visualizeFeatureMap(act);


	outputEight = model(sampleEight);
	#act = activation['conv2'].squeeze();
	#visualizeFeatureMap(act);
	act = activation['conv3'].squeeze();
	visualizeFeatureMap(act);


def shiftPixels(dataSample, numberToShift, direction):
	# direction=1 for left shift, -1 for right
	if direction==1:
		dataSample = np.pad(dataSample.squeeze(), ((0, 0), (0, 3)), 'edge');
		dataSample = [x[3:] for x in dataSample];
	elif direction==-1:
		dataSample = np.pad(dataSample.squeeze(), ((0, 0), (3, 0)), 'edge');
		dataSample = [x[:-3] for x in dataSample];
	
	dataSample = torch.FloatTensor(dataSample);
	print(dataSample.shape);
	return dataSample.view(1, 1, dataSample.shape[0], dataSample.shape[1]);

	
def shiftAndCheck(model, testLoader):
	
	digit = 1;
	(accurateOne, data) = findSpecificLabel(digit, model, testLoader);
	shiftedLeftData = shiftPixels(data[accurateOne], 3, 1);
	shiftedRightData = shiftPixels(data[accurateOne], 3, -1);
	
	plt.imshow(data[accurateOne].squeeze(), cmap='gray');
	plt.show();
	plt.imshow(shiftedLeftData.squeeze(), cmap='gray');
	plt.show();
	plt.imshow(shiftedRightData.squeeze(), cmap='gray');
	plt.show();

	print(model(data[accurateOne].unsqueeze(0)));
	shiftedLeftPrediction = model(shiftedLeftData);
	print(shiftedLeftPrediction);
	shiftedRightPrediction = model(shiftedRightData);
	print(shiftedRightPrediction);

	print(shiftedLeftPrediction.argmax(dim=1));
	print(shiftedRightPrediction.argmax(dim=1));

if __name__ == '__main__':
	trainBatchSize = 500;
	testBatchSize = 100;
	epochs = 5;

	trainLoader = torch.utils.data.DataLoader(
		datasets.MNIST('./data', train=True, download=True, 
				transform=transforms.Compose([
					transforms.ToTensor(),
					transforms.Normalize((0.1307,), (0.3081,))
					])),
		batch_size=trainBatchSize, shuffle=True);
	
	testLoader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=testBatchSize, shuffle=True);

	'''
	model = Network();
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01);

	
	#modelNew = models.alexnet(pretrained=True);
	#plotWeights(model.conv1);
	#print(model.conv1.weight.data);
	for epoch in range(0, epochs):
		train(model, trainLoader, optimizer, epoch);
		#validate(model, trainLoader);
		#test(model, testLoader);
	
	torch.save(model, 'MNIST_model.pt');
	'''
	model = torch.load('MNIST_model.pt');
	model.eval();

	#plotWeights(model.conv1);
	#print(model.conv1.weight.data);
	#plotWeights(model.conv2);
	#plotWeights(model.conv3);
	
	#plotFeatureMaps(model, testLoader);	
	
	shiftAndCheck(model, testLoader);
