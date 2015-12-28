import random,math

# pyNeural for Python 3.4 - Simple Neuron Class
# https://github.com/mauricioribeiro/pyNeural
class Neuron(object):

	def __init__(self,nWeights,transferFunction):
		#self.weights = [round(random.uniform(0,1),3) for w in range(nWeights)]
		self.weights = [random.uniform(0,1) for w in range(nWeights)]
		self.inputs = [0]*nWeights
		self.threshold = 0.5
		self.alpha = 0.1
		self.function = transferFunction
		self.gradientError = None

	def setInputs(self,inputArray):
		self.inputs = inputArray

	def setWeight(self,index,value):
		self.weights[index-1] = value

	def setWeights(self,weightsArray):
		self.weights = weightsArray

	def setThreshold(self,value):
		self.threshold = value

	def setLearningRate(self,value):
		self.alpha = value

	def setGradientError(self,value):
		self.gradientError = value

	def getWeight(self,index):
		return self.weights[index-1]

	def getWeights(self):
		return self.weights

	def getInput(self,index):
		return self.inputs[index-1]

	def getThreshold(self):
		return self.threshold

	def getLearningRate(self):
		return self.alpha

	def getTransferFunctions(self):
		return ['step','linear','sigmoid']

	def getTransferFunction(self):
		return self.function if self.function in self.getTransferFunctions() else 'Invalid Function'

	def getSum(self):
		u = 0
		for i in self.rangeWeights():
			u += self.getInput(i)*self.getWeight(i)
		return u

	def countInputs(self):
		return len(self.inputs)

	def rangeWeights(self):
		return range(1,len(self.weights)+1)

	def transferFunction(self,u):
		if self.function == 'step':
			return 1 if u > self.getThreshold() else 0
		if self.function == 'sign':
			r = 0
			if u > self.getThreshold(): r = 1
			if u < -self.getThreshold(): r = -1
			return r
		if self.function == 'linear':
			return u
		if self.function == 'sigmoid':
			return 1/(1+math.exp(-u))

	def think(self):
		return self.transferFunction(self.getSum())
