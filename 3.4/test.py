from neuron import Neuron
from backpropagation import BackPropagationNet

'''
	pyNeural for Python 3.4 - building Backpropagation network
	Making network example from https://www.youtube.com/watch?v=I2I5ztVfUSE [video]
'''

bias = 1
network = BackPropagationNet()
# createFromPattern() parameters: patternString , initialInputsByNeuron (without Bias inputs), flag if newtwork has bias
network.createFromPattern('2-1', 2, bias)
# setting Learning Rate (alpha) to 0.0001 [ https://youtu.be/I2I5ztVfUSE?t=11m ]
network.setLearningRate(0.0001)

# printing names of layers
print(network.getLayers())

if network.checkAllLayers():
	# setWeights() parameters: idLayer, index of Neuron (starting at 1), weightsArray (including Bias weight)
	network.setWeights('input_layer', 1, [0.1, -0.2, 0.1])
	network.setWeights('input_layer', 2, [0.1, -0.1, 0.3])
	network.setWeights('output_layer', 1, [0.2, 0.2, 0.3])

	# input array (without bias, therefore bias was already passed to the network, through the createFromPattern() function)
	inputs = [0.1, 0.9]
	# printing network output
	print(network.think(inputs))

	# printing v (neurons sum) and y (neurons output) values
	print(network.getNeuron('input_layer',1).getSum(), network.getNeuron('input_layer',1).think()) # v1, y1 [ https://youtu.be/I2I5ztVfUSE?t=12m50s ]
	print(network.getNeuron('input_layer',2).getSum(), network.getNeuron('input_layer',2).think()) # v2, y2 [ https://youtu.be/I2I5ztVfUSE?t=14m ]

	# training the network
	inputMatrix, targetOutput = [inputs],[0.9]
	network.setMaxInteractions(1)
	network.train(inputMatrix, targetOutput)

	# printing new calculated weights
	for neurons in network.rangeLayers():
		print('---')
		for neuron in neurons:
			print(neuron.getWeights()) # weights by neuron [video at 27:27s]