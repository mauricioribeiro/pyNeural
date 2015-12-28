# pyNeural
![pyNeural](https://github.com/mauricioribeiro/pyNeural/blob/master/pyneural.png "pyNeural")
<h3>Aritificial Neural Network classes (For Python 2.7 and 3.4)</h3>
<p>pyNeural is a simple set of classes for generating ANNs (<i>Artificial Neural Networks</i>) easily. It isn't done yet. If you have any suggestions, feel free  to become a contributor.</p>
<br>
##Main Functions (2.7)
<h4>Neuron Class</h4>
<p>This class is used for create an Artificial Neuron. It takes just two parameters:</p>
<ul>
	<li><b>nWeights [int]</b> the amount of weights (and respectively the inputs) of the neuron.</li>
	<li><b>transferFunction [string]</b> the transfer function used by the neuron. PS: you can check out all the transfer functions available calling <i>getTransferFunctions()</i> function.</li>
</ul>
<h5>setInputs(inputArray)</h5>
<ul>
	<li><b>inputArray [float List]</b> is an array of the inputs values (ps: the array must be same length as Neuron.weights length).</li>
	<li>It's a void function which sets the Neuron inputs.</li>
</ul>
<h5>setWeight(index,value)</h5>
<ul>
	<li><b>index [int]</b> is the weight position desired to change. PS: Beginning by 1.</li>
	<li><b>value [float]</b> is a new value of weight specified by index.</li>
	<li>It's a void function which sets the a specified weight.</li>
</ul>
<h5>setWeights(weightsArray)</h5>
<ul>
	<li><b>weightsArray [float List]</b> is an array of weight values.</li>
	<li>It's a void function which sets the Neuron weights. Just a reminder, the class Neuron receives as parameter the amount of weights (nWeights). After you initialize the class, you can set its weights or use the random generated weights.</li>
</ul>
<h5>setLearningRate(value)</h5>
<ul>
	<li><b>value [float]</b> is the value of the Learning Rate used in the training. This value is known as <i>alpha value</i> as well.</li>
	<li>It's a void function which sets the Neuron Learning Rate.</li>
</ul>
<h5>setMaxInteractions(highValue)</h5>
<ul>
	<li><b>highValue [int]</b> is the value of maximum interactions allowed for the Neuron try to converge.</li>
	<li>It's a void function which sets the the value of maximum interactions. By default, this value is 1000.</li>
</ul>
<h5>getInput(index)</h5>
<ul>
	<li><b>index [int]</b> is the input position desired to get. PS: Starting at 1.</li>
	<li>It returns the specified input value [float].</li>
</ul>
<h5>getWeight(index)</h5>
<ul>
	<li><b>index [int]</b> is the weight position desired to get. PS: Starting at 1.</li>
	<li>It returns the specified weight value [float].</li>
</ul>
<h5>getLearningRate()</h5>
<ul>
	<li>It returns the Learning Rate [float] used in the training.</li>
</ul>
<h5>getMaxInteractions()</h5>
<ul>
	<li>It returns the value of maximum interactions [int] used in the training.</li>
</ul>
<h5>getTransferFunction()</h5>
<ul>
	<li>It returns the Neuron Transfer Function [string]. It also can return "<i>Invalid Function</i>" string.</li>
</ul>
<h5>getTransferFunctions()</h5>
<ul>
	<li>It returns an array [string List] with the keys of the transfer functions available. Until the last commit, they are: <b>step</b>, <b>sign</b>, <b>linear</b> and <b>sigmoid</b>. <a href="https://en.wikipedia.org/wiki/Artificial_neuron#Types_of_transfer_functions" target="_blank">Read more about Transfer Functions here</a>.</li>
</ul>
<h5>rangeWeights()</h5>
<ul>
	<li>It returns the count of weights [List]. It can be used in the Python <i>for</i>.</li>
</ul>
<h5>train(inputMatrix,desiredArray)</h5>
<ul>
	<li><b>inputMatrix [List of float List]</b> is the input matrix values for training. Each array [float List] from this matrix represents the Neuron inputs. Rows represent the amount of samples and columns represent the inputs.</li>
	<li><b>desiredArray [float List]</b> is the desired outputs of inputMatrix. Note that if the inputMatrix is 5x3 (samples x inputs), desiredArray must be an array [float List] of length 5.</li>
	<li>It returns True or False, depending if the Neuron training converged [Boolean].</li>
</ul>
<h4>Neural Net Class</h4>
<p>This class is used for create a whole Neural Network. It isn't totally written yet.</p>
<br>
##Getting Started
<p>These are test files with basic examples about how the classes work:</p>
<ul>
	<li><a href="https://github.com/mauricioribeiro/pyNeural/blob/master/test-neuron.py" target="_blank">test-neuron.py</a> How create, setup and train an unique Neuron. It uses the class <b>Neuron</b>.</li>
	<li><a href="https://github.com/mauricioribeiro/pyNeural/blob/master/test-neuralnet.py" target="_blank">test-neuralnet.py (beta)</a> How create layers, add neurons and setup a whole Neural Network. It uses the classes <b>Neuron</b> and <b>NeuralNet (beta)</b>.</li>
</ul>
