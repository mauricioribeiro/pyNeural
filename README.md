# pyNeural
<h3>Aritificial Neural Network classes (Python 2.7)</h3>
<p>pyNeural is a simple set of classes for generate ANNs (<i>Artificial Neural Networks</i>) easily. It isn't done yet. Any suggestion? Ask for be a contributor.</p>
<br>
##Main Functions
<h5>setWeights(weightsArray)</h5>
<ul>
<li><b>weightsArray [float List]</b> is an array of weight values.</li>
<li>This function is a void function which sets the Neuron weights. Just to remember, the class Neuron receives as parameter the amount of weights (nWeights). After you initialize the class, you can set its weights or use the random weights generated.</li>
</ul>
<h5>setInputs(inputArray)</h5>
<ul>
<li><b>inputArray [float List]</b> is an array of inputs values (ps: the array must be same length as Neuron.weights length).</li>
<li>This function is a void function which sets the Neuron inputs.</li>
</ul>
<h5>setWeight(index,value)</h5>
<ul>
<li><b>index [int]</b> is the weight position desired to change. PS: Beginning by 1.</li>
<li><b>value [float]</b> is a new value of weight specified by index.</li>
<li>This function is a void function which sets the a specified weight.</li>
</ul>
<h5>getInput(index)</h5>
<ul>
<li><b>index [int]</b> is the input position desired to get. PS: Beginning by 1.</li>
<li>This function returns the specified input value [float].</li>
</ul>
<h5>getWeight(index)</h5>
<ul>
<li><b>index [int]</b> is the weight position desired to get. PS: Beginning by 1.</li>
<li>This function returns the specified weight value [float].</li>
</ul>
<h5>rangeWeights()</h5>
<ul>
<li>This function returns the count of weights [List]. It can be used in the Python <i>for</i>.</li>
</ul>
<h5>train(inputMatrix,desiredArray)</h5>
<ul>
<li><b>inputMatrix [List of float List]</b> is the input matrix values for training. Each array [float List] of this matrix represent the Neuron inputs. Rows represent the amount of samples and columns represent the inputs.</li>
<li><b>desiredArray [float List]</b> is the desired outputs of inputMatrix. Note that if the inputMatrix is 5x3 (samples x inputs), desiredArray must be a array [float List] of length 5.</li>
<li>This function returns True or False, depending if the Neuron training converged [Boolean].</li>
</ul>
<br>
##Getting Started
<p>There is a test file with a basic example about how an unique neuron works.</p>
