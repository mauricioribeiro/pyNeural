from neuron import Neuron

myNeuron = Neuron(3,'step')
myNeuron.setWeights([0,0,0])
myNeuron.setMaxInteractions(100)

trainingMatrix = [[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
desiredArray = [1,1,1,0]
converged = myNeuron.train(trainingMatrix,desiredArray)

if converged:
	print 'Training OK (%d interactions)' %(myNeuron.getTrainingInteractions())
else:
	print 'Training FAILED (+%d interactions)' %(myNeuron.getMaxInteractions())


for i in myNeuron.rangeWeights():
	print 'W'+str(i)+': '+str(myNeuron.getWeight(i))

myNeuron.setInputs([1,0,0])
print myNeuron.getTrainingInteractions()
print 'Output: '+str(myNeuron.think())