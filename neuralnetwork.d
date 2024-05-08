// This module contains tools for you to create a feed-forward neural network easily.

module neuralnetwork;

import std.math : abs;

// this variable will contain the learning rate of the network
float learningRate = 1.0;

// this function is the sigmoid function
float sigmoid(float x)
{
    // it returns the result of the x value through the sigmoid function
    return 1.0 / (1.0 + 2.7182818 ^^ -x);
}

// this is the derivative of the sigmoid function
float gradient(float x)
{
    // it returns the result of x through the derivative of the sigmoid function
    return x * (1.0 - x);
}

// this class will represent a synapse
class Synapse
{
    // this will be the weight of the synapse
    float weight;
    // these will be the neurons it connects, the one in front and the one behind
    Neuron backwardNeuron, forwardNeuron;

    // this is the constructor
    this (float weight, ref Neuron backwardNeuron, ref Neuron forwardNeuron)
    {
        this.weight = weight;
        this.forwardNeuron = forwardNeuron, this.backwardNeuron = backwardNeuron;
    }
}

// this class will represent a neuron
class Neuron
{
    // this will be the name of the neuron
    string name;
    // these will be the delta, the error (must start as 0.0), the input (must start as 0.0) and the output values
    float delta, error = 0.0, input = 0.0, output;
    // these will be the arrays with all forward and backward synapses
    Synapse[] backwardSynapses, forwardSynapses;

    // this is the constructor
    this (string name)
    {
        this.name = name;
    }

    // this member function will do the back-propagation, that makes the network learn by adjusting the weights of the synapses of the neuron
    void backpropagateAndAdjust()
    {
        // if it has forward synapses, meaning it is not the output layer
        if (this.forwardSynapses != [])
            // calculate the error as the mean of all errors
            this.error /= this.forwardSynapses.length;

        // if the error is very high
        if (abs(this.error) > 0.9)
            // make the delta also very high (this is to prevent neuron saturation when the gradient is too tiny)
            this.delta = this.error * 0.15;
        // if the error was reasonable
        else
            // calculate the delta
            this.delta = this.error * gradient(this.output);

        // set the error back to 0.0
        this.error = 0.0;

        // use a loop to go through all backward synapses
        foreach (ref synapse; this.backwardSynapses)
        {
            // send the error to the neuron in the previous layer
            synapse.backwardNeuron.error += this.delta * synapse.weight;
            // adjust the weight of the synapse with the Error Weighted Derivative formula
            synapse.weight += synapse.backwardNeuron.output * this.delta * learningRate;

            // if the weight has become too low
            if (synapse.weight < -20.0)
                // set it as -20.0, this is to prevent the weight from becoming so small that it ends up over weighting all the other synapses
                synapse.weight = -20.0;
            // if the weight has become too big
            else if (synapse.weight > 20.0)
                // set it as 20.0, this is to prevent the weight from becoming so big that it ends up over weighting all the other synapses
                synapse.weight = 20.0;
            // if it becomes a NaN value, this cannot be allowed
            else if (synapse.weight is float.nan || synapse.weight is -float.nan)
                // set it back to 0.0
                synapse.weight = 0.0;
        }
    }
}

// this struct will represent the entire network
class Network
{
    // this array will contain the layers, which in turn are formed by neurons
    Neuron[][] layers;

    // this is the constructor
    this (ref Neuron[][] layers)
    {
        this.layers = layers;
    }

    // this member function will do the thinking with the whole network, that produces the output
    float think(float[] input)
    {
        // use a loop to go through the neurons of the first layer, meaning the input layer
        foreach (i, ref neuron; this.layers[0])
        {
            // set the output as the input because in the first layer they are the same
            neuron.output = input[i];

            // use a loop to go through all synapses to feed-forward to the second layer
            foreach (ref synapse; neuron.forwardSynapses)
                // activate the next neuron
                synapse.forwardNeuron.input += neuron.output * synapse.weight;
        }

        // now we use a loop to go through all the other layers, doing the same we did above
        foreach (ref layer; this.layers[1 .. $])
            // use a loop to go through every neuron in the layer
            foreach (i, ref neuron; layer)
            {
                // calculate the output with the sigmoid function and set its input back to 0.0
                neuron.output = sigmoid(neuron.input), neuron.input = 0.0;

                // use a loop to go through all forward synapses to feed-forward to the next layer
                foreach (ref synapse; neuron.forwardSynapses)
                    // activate the next neuron
                    synapse.forwardNeuron.input += neuron.output * synapse.weight;
            }

        // return the output of the last layer, which is the output layer
        return this.layers[$ - 1][0].output;
    }

    // this member function will do the back-propagation with gradient descent, that produces the learning
    void learn(float error)
    {
        // set the error of the only output neuron, which is in the last layer
        this.layers[$ - 1][0].error = error;

        // use a loop to go through all layers, from last to first, except the input layer (the first layer)
        foreach_reverse (ref layer; this.layers[1 .. $])
            // use a loop to go through all neurons in the layer
            foreach (ref neuron; layer)
                // make the neuron call its 'backpropagateAndAdjust()' member function
                neuron.backpropagateAndAdjust();
    }
}
