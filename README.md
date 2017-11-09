# Swift Neural Network

Simple Swift implementation of Neural Network. Any size of network can be created.

![Alt text](NN.jpg?raw=true "Neural Network")

## Usage

Creates small neural network with four input nodes and two output nodes and two hidden layers with 7 and 5 nodes:
```swift
let nn = NeuralNetwork(layers: [4, 7, 5, 2], bias: 0.75, eta: 0.5)
```

Teaches network with input and expected output
```swift
nn.teach(input: data.input, expectedOutput: data.output)
```
Evaluates some input
```swift
nn.eval(input: [0.8, 0.75, 0.1, 0.01])
```

## To do
- make it work on multiple threads and cores
