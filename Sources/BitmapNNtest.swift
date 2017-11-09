//
//  BitmapNNtest.swift
//  NeuralNetwork
//
//  Created by Matej Ukmar on 08/08/2017.
//  Copyright © 2017 zenplus. All rights reserved.
//

import Foundation


class BitmapNNtest {
	
	static func run() {
		
		print("Starting Neural Network bitmap test")
		//Simple 2 by 2 bitmap
		//Neural Network shoul train and then recognize horizontal and vertical lines
		
		let nn = NeuralNetwork(layers: [4, 7, 5, 2], bias: 0.75, eta: 0.5)
		
		struct TrainingData {
			let input: [Double]
			let output: [Double]
		}
		
		
		var trainingData: [TrainingData] = []
		
		trainingData.append(TrainingData(input: [1,1,0,0], output: [1, 0]))  //horizontal line
		trainingData.append(TrainingData(input: [0,0,1,1], output: [1, 0]))  //horizontal line
		trainingData.append(TrainingData(input: [1,0,1,0], output: [0, 1]))  //vertical line
		trainingData.append(TrainingData(input: [0,1,0,1], output: [0, 1]))  //vertical line
		trainingData.append(TrainingData(input: [1,1,1,1], output: [0, 0]))
		trainingData.append(TrainingData(input: [1,1,1,0], output: [0, 0]))
		trainingData.append(TrainingData(input: [1,1,0,1], output: [0, 0]))
		trainingData.append(TrainingData(input: [1,0,1,1], output: [0, 0]))
		trainingData.append(TrainingData(input: [0,1,1,1], output: [0, 0]))
		trainingData.append(TrainingData(input: [0,0,0,0], output: [0, 0]))
		trainingData.append(TrainingData(input: [0,0,0,1], output: [0, 0]))
		trainingData.append(TrainingData(input: [0,0,1,0], output: [0, 0]))
		trainingData.append(TrainingData(input: [0,1,0,0], output: [0, 0]))
		trainingData.append(TrainingData(input: [1,0,0,0], output: [0, 0]))
		
		for i in 0..<10000 { //train 10000 times
			if (i%500 == 0) {
				print("teaching loop: \(i) of 10000")
			}
			for data in trainingData {
				nn.teach(input: data.input, expectedOutput: data.output)
				//print("test for: \(nn.nodes[nn.layers.count-1][0]), \(nn.nodes[nn.layers.count-1][1])")
			}
		}
		
		nn.eval(input: [1, 1, 0, 0]) // horizontal
		print("test for horizontal top: \(nn.outputNodesString)")
		
		nn.eval(input: [0, 0, 1, 1]) // horizontal
		print("test for horizontal bottom: \(nn.outputNodesString)")
		
		nn.eval(input: [1, 0, 1, 0]) // vertical
		print("test for vert left: \(nn.outputNodesString)")
		
		
		nn.eval(input: [0, 1, 0, 1]) // vertical
		print("test for vert right: \(nn.outputNodesString)")
		
		nn.eval(input: [0, 1, 1, 1]) // mixed, value s should be close to 0
		print("test for 0111: \(nn.outputNodesString)")
		
		
		nn.eval(input: [0.8, 0.75, 0.1, 0.01]) // close to vertical
		print("test for 8,75,1,01: \(nn.outputNodesString)")
		
		
		//		Compare saved and loaded data
		print("\n\n\n\n****** Original Neural data ******")
		nn.printOutData()
		let saveURL = URL(fileURLWithPath: "./nndata")
		nn.saveData(url: saveURL)
		let nn2 = NeuralNetwork(dataUrl: saveURL)
		print("\n\n\n\n****** printing out loaded NN ******")
		print("It should be same as original data")
		nn2.printOutData()
		
		
	}
	
	
}
