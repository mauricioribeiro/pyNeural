from neuron import Neuron

# Simple Neural Network class
# https://github.com/mauricioribeiro/pyNeural
class NeuralNet(Neuron):

	def __init__(self):
		self.layers = {}

	def getLayerIds(self):
		r = []
		for l in self.layers:
			r.append(l)
		return r

	def getNeuron(self,idLayer,index):
		if self.checkLayer(idLayer):
			if index > 0 and index <= self.countNeurons(idLayer):
				return self.layers[idLayer]['neurons'][index-1]
			else:
				print 'NEURON NET SAYS: index must be a integer value between 1 and %d' %(self.countNeurons(idLayer))
		return False

	def addLayer(self,idLayer,neuronsArray,parentLayerId = None):
		self.layers[idLayer] = {
		'neurons': neuronsArray,
		'parent': parentLayerId
		}

	def addNeuron(self,idLayer,neuron):
		if type(neuron) is not Neuron:
			print 'NEURON NET SAYS: The neuron must be an instance of Neuron'
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

	def checkAllLayers(self):
		for l in self.layers:
			if self.self.layers[l]['parent']==None:
				if not self.checkLayer(self.layers[l]['parent']):
					return False
			# continue...

	def train(self):
		print 'NEURAL NET SAYS: training..'

	def think(self):
		print 'NEURAL NET SAYS: thinking..'
