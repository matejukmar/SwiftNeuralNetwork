//
//  NeuralNetwork.swift
//  NeuralNetwork
//
//  Created by Matej Ukmar on 11/04/2017.
//  Copyright © 2017 zenplus. All rights reserved.
//

import Foundation

extension Float {
	
	static var random: Float {
		return Float(arc4random()) / 0xFFFFFFFF
	}
	
	static func random(min: Float, max: Float) -> Float {
		return Float.random * (max - min) + min
	}
	
	
}

final class NeuralNetworkF {
	
	let layers: [Int]
	let bias: Float
	let eta: Float
	let euler: Float = Float(M_E)
	
	let verbose: Bool
	

	
	private var nodes: [[Float]] = []
	private var incomingWeights : [[[Float]]] = []  //'incoming' indicates the way arrays are structured. Each node of (next) layer has count(nodes from previpus layer) incoming waights
	private var biasWeights : [[Float]] = []

	
	
	
	init(layers: [Int], bias: Float, eta: Float, verbose: Bool = false) {
		self.verbose = verbose
		self.layers = layers
		self.bias = bias
		self.eta = eta
		initializeData()
	}
	
	
	init(dataUrl: URL, verbose: Bool = false) {
		
		self.verbose = verbose
		
		let data = try! Data(contentsOf: dataUrl)
		
		var start = 0
		
		let etaData = data.subdata(in: start..<(start+MemoryLayout<Float>.size))
		let eta: Float = etaData.withUnsafeBytes { (ptr: UnsafePointer<Float>) -> Float in
			return ptr.pointee
		}
		self.eta = eta
		
		start += MemoryLayout<Float>.size
		
		let biasData = data.subdata(in: start..<(start+MemoryLayout<Float>.size))
		let bias: Float = biasData.withUnsafeBytes { (ptr: UnsafePointer<Float>) -> Float in
			return ptr.pointee
		}
		self.bias = bias
		
		start += MemoryLayout<Float>.size
		
		let lcData = data.subdata(in: start..<(start+MemoryLayout<Int>.size))
		let layerCount: Int = lcData.withUnsafeBytes { (ptr: UnsafePointer<Int>) -> Int in
			return ptr.pointee
		}
		
		start += MemoryLayout<Int>.size
		
		var layers: [Int] = []
		for _ in 0..<layerCount {
			let numData = data.subdata(in: start..<(start+MemoryLayout<Int>.size))
			let num: Int = numData.withUnsafeBytes { (ptr: UnsafePointer<Int>) -> Int in
				return ptr.pointee
			}
			layers.append(num)
			start += MemoryLayout<Int>.size
		}
		self.layers = layers
		
		var incomingWeights: [[[Float]]] = []
		for layerIndex in 1..<layers.count {
			var layerWeights: [[Float]] = []
			for _ in 0..<layers[layerIndex] {
				var nodeWeights: [Float] = []
				for _ in 0..<layers[layerIndex-1] {
					
					let weightData = data.subdata(in: start..<(start+MemoryLayout<Float>.size))
					let weight: Float = weightData.withUnsafeBytes { (ptr: UnsafePointer<Float>) -> Float in
						return ptr.pointee
					}
					nodeWeights.append(weight)
					start += MemoryLayout<Float>.size
				}
				layerWeights.append(nodeWeights)
			}
			incomingWeights.append(layerWeights)
		}
		self.incomingWeights = incomingWeights
		
		var biasWeights: [[Float]] = []
		for layerIndex in 1..<layers.count-1 {
			var layerBiasWeights: [Float] = []
			for _ in 0..<layers[layerIndex] {
				let weightData = data.subdata(in: start..<(start+MemoryLayout<Float>.size))
				let weight: Float = weightData.withUnsafeBytes { (ptr: UnsafePointer<Float>) -> Float in
					return ptr.pointee
				}
				layerBiasWeights.append(weight)
				start += MemoryLayout<Float>.size
			}
			biasWeights.append(layerBiasWeights)
		}
		
		self.biasWeights = biasWeights
		
	}
	
	

	func initializeData() {
		
		let startDate = Date()
		
		var numNodes = 0
		var numWeights = 0
		var numBiasWeights = 0

		
		//Fill all nodes with 0s
		for i in 0..<layers.count {
			var layerNodes: [Float] = []
			for _ in 0..<layers[i] {
				layerNodes.append(0)
				numNodes += 1
			}
			nodes.append(layerNodes)
		}
		
		
		//Set up random incomingWeights
		for i in 1..<layers.count { //starts with 1 because input nodes do not have incoming weights
			var layerWeights: [[Float]] = []
			for _ in 0..<layers[i] {
				var nodeWeights: [Float] = []
				for _ in 0..<layers[i-1] {
					nodeWeights.append(Float.random(min: -1, max: 1))
					numWeights += 1
				}
				layerWeights.append(nodeWeights)
			}
			incomingWeights.append(layerWeights)
		}
		
		
		//Set up bias weights
		for i in 1..<layers.count-1 {  //input and output nodes do not have bias weights
			var biasLayerWeights: [Float] = []
			for _ in 0..<layers[i] {
				biasLayerWeights.append(Float.random(min: -1, max: 1))
				numBiasWeights += 1
			}
			biasWeights.append(biasLayerWeights)
		}
		
		let endDate = Date()
		if verbose {
			print("created \(numNodes) nodes, \(numWeights) weights, \(numBiasWeights) bias weights")
			print("data initialization time \(endDate.timeIntervalSince(startDate)) seconds")
		}
	}
	
	func sigmoid(x: Float) -> Float {
		return 1/(1+powf(euler, -x))
	}
	
	
	func floatArrayString(array: [Float]) -> String {
		var result = "["
		var first = true
		for elm in array {
			if !first {
				result.append(",")
			} else {
				first = false
			}
			result.append("\(elm)")
		}
		result.append("]")
		return result
	}

	
	
	func saveData(url: URL) {
		var data = Data()
		var theEta = eta
		data.append(UnsafeBufferPointer(start: &theEta, count: 1))
		var theBias = bias
		data.append(UnsafeBufferPointer(start: &theBias, count: 1))
		var layerCount = layers.count
		data.append(UnsafeBufferPointer(start: &layerCount, count: 1))
		
		for val in layers {
			var theVal = val
			data.append(UnsafeBufferPointer(start: &theVal, count: 1))
		}
		
		for layerIndex in 0..<incomingWeights.count {
			for nodeIndex in 0..<incomingWeights[layerIndex].count {
				for incomingWeightIndex in 0..<incomingWeights[layerIndex][nodeIndex].count {
					var weight = incomingWeights[layerIndex][nodeIndex][incomingWeightIndex]
					data.append(UnsafeBufferPointer(start: &weight, count: 1))
				}
			}
		}
		
		for layerIndex in 0..<biasWeights.count {
			for nodeIndex in 0..<biasWeights[layerIndex].count {
				var weight = biasWeights[layerIndex][nodeIndex]
				data.append(UnsafeBufferPointer(start: &weight, count: 1))
			}
		}
		
		try! data.write(to: url)
		
	}
	
	func eval(input: [Float]) {
		forwardPropagate(input: input)
	}
	
	func forwardPropagate(input: [Float]) {
		
		guard input.count == layers[0] else {
			print("incorrect length of input array")
			return
		}
		
		let startDate = Date()
		
		nodes[0] = input
		
		// forward propagation
		
		for nextLayerIndex in 1..<layers.count {
			let prevLayerIndex = nextLayerIndex-1
			for nextLayerNodeIndex in 0..<nodes[nextLayerIndex].count {
				var sum: Float = 0
				//sum incoming weights for each node in layer
				for prevLayerNodeIndex in 0..<nodes[prevLayerIndex].count {
					sum += nodes[prevLayerIndex][prevLayerNodeIndex]*incomingWeights[prevLayerIndex][nextLayerNodeIndex][prevLayerNodeIndex]
				}
				
				if nextLayerIndex == layers.count-1 { //we do not sigmoid for last layer
					//assign value to node
					nodes[nextLayerIndex][nextLayerNodeIndex] = sigmoid(x: sum)
				} else {
					
					//add bias value for node
					sum += biasWeights[nextLayerIndex-1][nextLayerNodeIndex]*bias
					
					//sigmoiding value
					let sigmoidSum = sigmoid(x: sum)
					
					//assign value to node
					nodes[nextLayerIndex][nextLayerNodeIndex] = sigmoidSum
				}
			}
		}
		let endDate = Date()
		
		if verbose {
			print("forward prop: \(floatArrayString(array: nodes[layers.count-1]))")
			print("forward propagation time \(endDate.timeIntervalSince(startDate)) seconds")
		}
	}
	
	func backwardPropagate(expectedOutput: [Float]) {
		
		guard expectedOutput.count == layers[layers.count-1] else {
			print("incorrect length of output array")
			return
		}
		
		// backward propagation
		
		let startDate = Date()
		
		var downstreamDeltas: [Float] = []
		var currentDeltas: [Float] = []
		
		for layerIndex in (0..<layers.count).reversed() {
			currentDeltas.removeAll()
			for nodeIndex in 0..<nodes[layerIndex].count {
				if layerIndex == layers.count-1 { //Ouput nodes
					let output = nodes[layerIndex][nodeIndex]
					let expOutput = expectedOutput[nodeIndex]
					let delta = output*(1-output)*(output-expOutput)
					currentDeltas.append(delta)
				} else {
					
					let output = nodes[layerIndex][nodeIndex]
					var downDletaSum: Float = 0
					for downstreamNodeIndex in 0..<nodes[layerIndex+1].count {
						let downDelta = downstreamDeltas[downstreamNodeIndex]
						let weight = incomingWeights[layerIndex][downstreamNodeIndex][nodeIndex]
						downDletaSum += downDelta*weight
					}
					let delta = output*(1-output)*downDletaSum
					currentDeltas.append(delta)
					
					for downstreamNodeIndex in 0..<nodes[layerIndex+1].count {
						let downDelta = downstreamDeltas[downstreamNodeIndex]
						let deltaWeight = (-eta)*downDelta*output
						//print("deltaWeight: \(deltaWeight)")
						incomingWeights[layerIndex][downstreamNodeIndex][nodeIndex] += deltaWeight
					}
					
					if layerIndex > 0 {
						let biasDelta = (-eta)*delta
						//print("biasDelta: \(biasDelta)")
						biasWeights[layerIndex-1][nodeIndex] += biasDelta
					}
					
				}
			}
			downstreamDeltas = currentDeltas
		}
		let endDate = Date()
		
		if verbose {
			print("backward propagation time \(endDate.timeIntervalSince(startDate)) seconds")
		}
		
	}
	
	func teach(input: [Float], expectedOutput: [Float]) {
		guard input.count == layers[0] else {
			print("incorrect length of input array")
			return
		}
		guard expectedOutput.count == layers[layers.count-1] else {
			print("incorrect length of output array")
			return
		}
		forwardPropagate(input: input)
		
		if verbose {
			var totalError: Float = 0
			for i in 0..<layers[layers.count-1] {
				totalError += powf(expectedOutput[i]-nodes[layers.count-1][i], 2)/2
			}
			print("total error: \(totalError)")
		}
		
		backwardPropagate(expectedOutput: expectedOutput)
	}
	
	
	func printOutData() {
		var data = ""
		data.append("\(eta),")
		data.append("\(bias),")
		data.append("\(layers.count):")
		
		for val in layers {
			data.append("\(val),")
		}
		
		data.append("\nWeightData\n")
		
		for layerIndex in 0..<incomingWeights.count {
			for nodeIndex in 0..<incomingWeights[layerIndex].count {
				for incomingWeightIndex in 0..<incomingWeights[layerIndex][nodeIndex].count {
					let weight = incomingWeights[layerIndex][nodeIndex][incomingWeightIndex]
					data.append("\(weight),")
				}
			}
		}

		data.append("\nBiasWeightData\n")

		for layerIndex in 0..<biasWeights.count {
			for nodeIndex in 0..<biasWeights[layerIndex].count {
				let weight = biasWeights[layerIndex][nodeIndex]
				data.append("\(weight),")
			}
		}
		
		print(data)
		
	}
	
	var outputNodesString: String {
		return floatArrayString(array: nodes[layers.count-1])
	}
	
	
}
