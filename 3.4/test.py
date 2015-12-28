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

print(network.getLayers())

if network.checkAllLayers():
	# setWeights() parameters: idLayer, index of Neuron (starting at 1), weightsArray (including Bias weight)
	network.setWeights('input_layer', 1, [0.1, -0.2, 0.1])
	network.setWeights('input_layer', 2, [0.1, -0.1, 0.3])
	network.setWeights('output_layer', 1, [0.2, 0.2, 0.3])

	inputs = [0.1, 0.9]
	network.setInputs('input_layer',inputs)
	print(network.getNeuron('input_layer',1).getSum(), network.getNeuron('input_layer',1).think()) # v1, y1 [video at 12:50s]
	print(network.getNeuron('input_layer',2).getSum(), network.getNeuron('input_layer',2).think()) # v2, y2 [video at 14:00s]
	print(network.think(inputs))