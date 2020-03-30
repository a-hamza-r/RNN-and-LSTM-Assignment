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


def findSpecificLabel(digit, model, testLoader):
	for _, (data, target) in enumerate(testLoader):
		output = model(data);
		pred = output.argmax(dim=1);
		allDigits = (pred == digit).nonzero().squeeze();
		accurateDigit = torch.tensor([x for x in allDigits if target[x] == digit], dtype=torch.int32)[0];
		return (accurateDigit, data);


def occlude(model, data, label):
	blackPixel = -0.4242;
	occludeSize = 8;
	occludeStride = 1;
	print(data.shape);
	
	height = data.shape[0];
	width = data.shape[1];
	
	outputHeight = int(np.ceil((height-occludeSize)/occludeStride));
	outputWidth = int(np.ceil((width-occludeSize)/occludeStride));
	
	probMap = torch.zeros((outputHeight, outputWidth));
	highestProbMap = torch.zeros((outputHeight, outputWidth));
	labelMap = torch.zeros((outputHeight, outputWidth));
	
	for row in range(height):
		for col in range(width):
			colStart, rowStart = col*occludeStride, row*occludeStride;
			colEnd, rowEnd = min(width, colStart+occludeSize), min(height, rowStart+occludeSize);

			if rowEnd >= height or colEnd >= width:
				continue;

			dataCopy = data.clone();

			dataCopy[rowStart:rowEnd, colStart:colEnd] = blackPixel;
			
			pred = model(dataCopy.unsqueeze(0).unsqueeze(0));
			
			probMap[row, col] = pred.squeeze()[6].detach();

			highestProbMap[row, col] = torch.max(pred, dim=1).values.detach();

			labelMap[row, col] = pred.argmax(dim=1);
			# print(pred.argmax(dim=1));
	print(labelMap);
	plt.imshow(data, cmap='gray');
	plt.show();
	
	plt.imshow(probMap, cmap='gray');
	plt.show();
	plt.imshow(highestProbMap, cmap='gray');
	plt.show();
	plt.imshow(labelMap, cmap='gray');
	plt.show();


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
	
	model = torch.load('MNIST_model.pt');
	model.eval();
	#test(model, testLoader);

	(accurateSix, data) = findSpecificLabel(3, model, testLoader);
	#print(data);
	#plt.imshow(data[accurateSix].squeeze(), cmap='gray');
	#plt.show();

	#plt.imshow([[-0.4242, -0.4242], [-0.4242, -0.4242]], cmap='gray');
	#plt.show();

	occlude(model, data[accurateSix].squeeze(), 3);
