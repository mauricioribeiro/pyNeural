import random,math

# Simple Neuron Class
# https://github.com/mauricioribeiro/pyNeural
class Neuron(object):

	def __init__(self,nWeights,transferFunction):
		self.weights = [round(random.uniform(0,1),3) for w in range(nWeights)]
		self.inputs = [0]*nWeights
		self.threshold = 0.5
		self.alpha = 0.1
		self.trainingInteractions = 0
		self.maxInteractions = 1000
		self.function = transferFunction

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

	def setMaxInteractions(self,highValue):
		self.maxInteractions = highValue if highValue > 0 else 1000

	def getWeight(self,index):
		return self.weights[index-1]

	def getInput(self,index):
		return self.inputs[index-1]

	def getThreshold(self):
		return self.threshold

	def getLearningRate(self):
		return self.alpha

	def getTransferFunctions(self):
		return ['step','linear']

	def getTransferFunction(self):
		return self.function if self.function in self.getTransferFunctions() else 'Invalid Function'

	def getMaxInteractions(self):
		return self.maxInteractions

	def getTrainingInteractions(self):
		return self.trainingInteractions

	def getSum(self):
		u = 0
		for i in self.rangeWeights():
			u += self.getInput(i)*self.getWeight(i)
		return u

	def addTrainingInteraction(self):
		self.trainingInteractions += 1

	def rangeWeights(self):
		return range(1,len(self.weights)+1)

	def transferFunction(self,u):
		if self.function == 'step':
			return 1 if u > self.getThreshold() else 0
		if self.function == 'linear':
			return u
		if self.function == 'sigmoid':
			return 1/(1+math.exp(-u))

	def checkAll(self):
		if self.getTransferFunction() not in self.getTransferFunctions():
			print 'NEURON SAYS: Invalid Transfer Function. Options available: '+str(self.getTransferFunctions())
			return False
		if self.getMaxInteractions() < 0:
			print 'NEURON SAYS: The Max Interactions must be greater than zero'
			return False
		return True

	def train(self,inputMatrix,desiredArray):
		if self.checkAll():
			while True:
			    errorCount,p = 0,0
			    for arrayInputs in inputMatrix:
			    	self.setInputs(arrayInputs)
			    	u = self.getSum()
			    	error = desiredArray[p]-self.transferFunction(u)
			    	if error != 0:
			    		errorCount += 1
			    		for x in self.rangeWeights():
			    			newWeight = self.getWeight(x)+self.getLearningRate()*error*self.getInput(x)
			    			self.setWeight(x,newWeight)
			    	p += 1
			    self.addTrainingInteraction()
			    if errorCount == 0: return True
			    if self.getTrainingInteractions() > self.getMaxInteractions(): return False

	def think(self):
		if self.checkAll():
			u = self.getSum()
			return self.transferFunction(u)
