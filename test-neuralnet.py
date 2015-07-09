from neuralnet import NeuralNet,Neuron

# neurons from First Layer
neuronA = Neuron(3,'step')
neuronB = Neuron(3,'step')
neuronC = Neuron(3,'step')
# neurons from Second Layer
neuronD = Neuron(2,'linear')

myNet = NeuralNet()
myNet.addLayer('firstLayer',[neuronA,neuronB])
myNet.addLayer('secondLayer',[neuronD])

myNet.addNeuron('firstLayer',neuronC)

print 'Amount of Layers:',myNet.countLayers()
print 'Amount of Neurons:',myNet.countNeurons()
print 'Amount of Neurons in "firstLayer"',myNet.countNeurons('firstLayer')
