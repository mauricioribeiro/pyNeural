from neuron import Neuron

# Simple Neural Network class
# https://github.com/mauricioribeiro/pyNeural
class NeuralNet(Neuron):

	def __init__(self):
		self.layers = {}

	def addLayer(self,idLayer,neuronsArray,parentLayerId = None):
		self.layers[idLayer]['neurons'] = neuronsArray
		self.layers[idLayer]['parent'] = parentLayerId
