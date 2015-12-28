from neuron import Neuron

# pyNeural for Python 3.4 - Simple Backpropagation Network class
# https://github.com/mauricioribeiro/pyNeural

'''
	UNIMPLEMENTED METHODS:
	- train()
'''
class BackPropagationNet(Neuron):

	def __init__(self):
		self.layers = {}
		self.threshold = 0.5
		self.alpha = 0.1
		self.bias = False

	def setInputs(self,idLayer,inputArray):
		if self.bias is not False:
			inputArray.insert(0, self.getBias())
		if self.checkLayer(idLayer):
			for n in range(len(self.layers[idLayer]['neurons'])):
				if self.layers[idLayer]['neurons'][n].countInputs() == len(inputArray):
					self.layers[idLayer]['neurons'][n].setInputs(inputArray)
	
	def setThreshold(self,value):
		self.threshold = value

	def setLearningRate(self,value):
		self.alpha = value
	
	def setBias(self,value):
		self.bias = value

	def setWeights(self,idLayer,index,weightsArray):
		if self.checkLayer(idLayer):
			self.layers[idLayer]['neurons'][index-1].setWeights(weightsArray)

	def getThreshold(self):
		return self.threshold

	def getLearningRate(self):
		return self.alpha

	def getBias(self):
		return self.bias if self.bias is not False else 0
	
	def getWeights(self,idLayer,index):
		if self.checkLayer(idLayer):
			return self.layers[idLayer]['neurons'][index-1].getWeights()

	def createFromPattern(self,patternString,initialInputsByNeuron,biasValue = False):
		layers, biasInput = patternString.split('-'), biasValue if biasValue is not False else 0
		previousInputs, parentId, lenLayers = initialInputsByNeuron + biasInput, None, len(layers)
		if lenLayers:
			for i in range(lenLayers):
				self.addLayer('hidden_%d' %i,self.generateNeurons(int(layers[i]),'sigmoid',previousInputs),parentId)
				previousInputs, parentId = int(layers[i]) + biasInput, 'hidden_%d' %i
			self.renameLayer('hidden_0','input_layer')
			self.renameLayer('hidden_%d' %(lenLayers-1),'output_layer')
		self.bias = biasValue

	def getLayerIds(self):
		r = []
		for l in self.layers:
			r.append(l)
		return r

	def getLayers(self):
		return self.layers

	def getLayer(self,idLayer):
		return self.layers[idLayer] if self.checkLayer(idLayer) else {}

	def getNeuron(self,idLayer,index):
		if self.checkLayer(idLayer):
			if index > 0 and index <= self.countNeurons(idLayer):
				return self.layers[idLayer]['neurons'][index-1]
		return False

	def getLayerSequence(self):
		r, target = [], None
		if self.checkAllLayers():
			lenLayers = len(self.layers)
			while len(r) < lenLayers:
				for l in self.layers:
					if self.layers[l]['parent'] == target:
						r.append(l)
						target = l
						break
		return r

	def generateNeurons(self,amount,transferFunction,previousInputs):
		return [Neuron(previousInputs,transferFunction) for i in range(amount)] if amount else []

	def addLayer(self,idLayer,neuronsArray,parentLayerId = None):
		self.layers[idLayer] = {
			'neurons': neuronsArray,
			'parent': parentLayerId
		}

	def addNeuron(self,idLayer,neuron):
		if type(neuron) is not Neuron:
			return False
		if self.checkLayer(idLayer):
			self.layers[idLayer]['neurons'].append(neuron)
			return True
		return False

	def renameLayer(self,idLayer,newIdLayer):
		if idLayer != newIdLayer and self.checkLayer(idLayer) and not self.checkLayer(newIdLayer):
			self.layers[newIdLayer] = self.layers.pop(idLayer)
			for l in self.layers.keys():
				if self.layers[l]['parent'] == idLayer:
					self.layers[l]['parent'] = newIdLayer
			return True
		return False

	def countLayers(self):
		return len(self.layers)

	def countNeurons(self,idLayer = False):
		if idLayer:
			if self.checkLayer(idLayer):
				return len(self.layers[idLayer]['neurons'])
		else: 
			r = 0
			for layer in self.layers.values():
				r += len(layer['neurons'])
			return r

	def checkLayer(self,idLayer):
		return True if idLayer in self.layers.keys() else False

	def checkAllLayers(self):
		for layer in self.layers.values():
			if layer['parent'] != None and not self.checkLayer(layer['parent']):
				return False
		return True

	def train(self):
		print('training...')

	def think(self,inputArray):
		layers, nextInputs  = self.getLayerSequence(), inputArray
		for l in layers:
			self.setInputs(l,nextInputs)
			nextInputs = []
			for n in range(len(self.layers[l]['neurons'])):
				nextInputs.append(self.layers[l]['neurons'][n].think())
		return nextInputs if len(nextInputs) > 1 else nextInputs[0]