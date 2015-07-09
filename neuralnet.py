from neuron import Neuron

# Simple Neural Network class
# https://github.com/mauricioribeiro/pyNeural
class NeuralNet(Neuron):

	def __init__(self):
		self.layers = {}

	def addLayer(self,idLayer,neuronsArray,parentLayerId = None):
		self.layers[idLayer] = {
		'neurons': neuronsArray,
		'parent': parentLayerId
		}

	def addNeuron(self,idLayer,neuron):
		if type(neuron) is not Neuron:
			print 'NEURON NET SAYS: The neuron must be a instance of Neuron'
			return False
		if self.checkLayer(idLayer):
			self.layers[idLayer]['neurons'].append(neuron)
			return True
		return False

	def checkLayer(self,idLayer):
		if self.layers.has_key(idLayer):
			return True
		print 'NEURAL NET SAYS: Invalid layer'
		return False

	def countLayers(self):
		return len(self.layers)

	def countNeurons(self,idLayer = False):
		if idLayer:
			if self.checkLayer(idLayer):
				return len(self.layers[idLayer]['neurons'])
		else: 
			r = 0
			for l in self.layers:
				r += len(self.layers[l]['neurons'])
			return r

	def train(self):
		print 'NEURAL NET SAYS: training..'
