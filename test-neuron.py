from neuron import Neuron

myNeuron = Neuron(3,'step')
myNeuron.setWeights([0,0,0]) # the first weight is of Bias
myNeuron.setMaxInteractions(100)

# learning the AND function
bias = 1
trainingMatrix = [[bias,0,0],[bias,1,0],[bias,1,1]]
desiredArray = [0,0,1]
converged = myNeuron.train(trainingMatrix,desiredArray)

if converged:
	print 'Training OK (%d interactions)' %(myNeuron.getTrainingInteractions())
else:
	print 'Training FAILED (+%d interactions)' %(myNeuron.getMaxInteractions())


for i in myNeuron.rangeWeights():
	print 'W'+str(i)+': '+str(myNeuron.getWeight(i))

myNeuron.setInputs([bias,0,1])
print myNeuron.getTrainingInteractions()
print 'Output: '+str(myNeuron.think()) #output desired is 0