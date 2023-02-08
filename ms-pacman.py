'''
=====================================================================================
Name        : Ms Pacman AI
Author      : Kenshiro
Version     : 1.00
Copyright   : GNU General Public License (GPLv3)
Description : Ms Pacman AI based on the T-Rex evolutionary neural network
=====================================================================================
'''

#Action space: Discrete(9)
#Observation space: Box(0, 255, (128,), uint8)
#Observation high: 255, low: 0

import gymnasium as gym
import math
import array

from ctypes import *
so_file = "/usr/local/lib/libT-Rex.so"
tRex = CDLL(so_file)

env = gym.make("ALE/MsPacman-v5", repeat_action_probability=0, obs_type="ram")
observation, info = env.reset()

#OBSERVATION DATA
RAM_BYTES = 128
RAM_BITS = RAM_BYTES * 8

#ACTION DATA
BITS_PER_ACTION = 4

#NEURAL NETWORK DATA
NEURAL_NETWORK_NUMBER_OF_INPUTS = RAM_BITS
NEURAL_NETWORK_NUMBER_OF_HIDDEN_LAYERS = 2
NEURAL_NETWORK_NUMBER_OF_OUTPUTS = BITS_PER_ACTION

#Number of consecutive victories to consider the training completed
MAX_CONSECUTIVE_VICTORIES = NEURAL_NETWORK_NUMBER_OF_INPUTS * NEURAL_NETWORK_NUMBER_OF_HIDDEN_LAYERS + NEURAL_NETWORK_NUMBER_OF_OUTPUTS

NEURAL_NETWORK_FILE_LOAD_ERROR = -10
NEURAL_NETWORK_ERROR_CODE = -12

NEURAL_NETWORK_FILE_PATH = create_string_buffer(str.encode("neural_network.json"))

action = env.action_space.sample() 

def computeNeuralNetworkInput(myObservation, myNeuralNetwork):

	bitArray = array.array('B',(0 for i in range(0, NEURAL_NETWORK_NUMBER_OF_INPUTS)))

	returnValue = 0
	bitArrayIndex = 0
	neuralNetworkInputIndex = 0

	for i in range(0, RAM_BYTES):
		for j in range (0, 7):

			bit = myObservation[i] >> j;

			if (bit & 1):
				bitArray[bitArrayIndex] = 1
			else:
				bitArray[bitArrayIndex] = 0
				
			bitArrayIndex += 1				

	while neuralNetworkInputIndex < NEURAL_NETWORK_NUMBER_OF_INPUTS and returnValue==0:
	
		returnValue = tRex.setNeuralNetworkInput(myNeuralNetwork, neuralNetworkInputIndex, bitArray[neuralNetworkInputIndex])

		neuralNetworkInputIndex += 1

	return returnValue
	
def parseNeuralNetworkOutput(neuralNetworkOutput, numberOfOutputs):

	action = 0
	
	neuralNetworkOutputIndex = 0
	
	while neuralNetworkOutputIndex < numberOfOutputs:

		if (neuralNetworkOutput[neuralNetworkOutputIndex]==1):
			action += pow(2, neuralNetworkOutputIndex)

		neuralNetworkOutputIndex += 1

	if action > 8:
		action = 0
	
	return action			

#Main logic
neuralNetworkFileFound = False

bestNeuralNetwork = c_void_p()
neuralNetworkClone = c_void_p()
auxNeuralNetwork = c_void_p()

#Check if a trained neural network already exists
returnValue = tRex.loadNeuralNetwork(NEURAL_NETWORK_FILE_PATH, byref(bestNeuralNetwork))

if returnValue==0:
	neuralNetworkFileFound = True
	print("Neural network file found")
elif returnValue==NEURAL_NETWORK_FILE_LOAD_ERROR:
	returnValue = tRex.createNeuralNetwork(byref(bestNeuralNetwork), NEURAL_NETWORK_NUMBER_OF_INPUTS, NEURAL_NETWORK_NUMBER_OF_HIDDEN_LAYERS, NEURAL_NETWORK_NUMBER_OF_OUTPUTS)

if returnValue==0:
	returnValue = tRex.createNeuralNetwork(byref(neuralNetworkClone), NEURAL_NETWORK_NUMBER_OF_INPUTS, NEURAL_NETWORK_NUMBER_OF_HIDDEN_LAYERS, NEURAL_NETWORK_NUMBER_OF_OUTPUTS)

consecutiveVictories = 0
bestReward = 0
episode = 0

while consecutiveVictories < MAX_CONSECUTIVE_VICTORIES and returnValue==0 and not neuralNetworkFileFound:
	
	observation, info = env.reset()

	reward = 0;
	episode += 1

	while returnValue==0:
		c_ubyte_p = POINTER(c_ubyte)
		myOutputArray = c_ubyte_p()
		myOutputArraySize = c_int()

		#print(observation)

		returnValue = computeNeuralNetworkInput(observation, neuralNetworkClone)

		if returnValue==0:
			returnValue = tRex.computeNeuralNetworkOutput(neuralNetworkClone, byref(myOutputArray), byref(myOutputArraySize))

		if returnValue==0 and myOutputArraySize.value!=NEURAL_NETWORK_NUMBER_OF_OUTPUTS:
			returnValue = NEURAL_NETWORK_ERROR_CODE

		if returnValue==0:
			action = parseNeuralNetworkOutput(myOutputArray, myOutputArraySize.value)

		observation, newReward, terminated, truncated, info =  env.step(action)		

		reward += newReward

		print("Reward:", reward)
		print("Best reward:", bestReward)
		print("Action:", action)
		print("Episode:", episode)

		if terminated or truncated:

			if reward > bestReward:
				
				auxNeuralNetwork = bestNeuralNetwork
				bestNeuralNetwork = neuralNetworkClone
				neuralNetworkClone = auxNeuralNetwork

				consecutiveVictories = 0
				bestReward = reward
			else:
				consecutiveVictories += 1

			returnValue = tRex.cloneNeuralNetwork(bestNeuralNetwork, neuralNetworkClone)

			if returnValue==0:
				returnValue = tRex.mutateNeuralNetwork(neuralNetworkClone)
			
			print("Episode finished")
			break

		print("Consecutive victories:", consecutiveVictories, "/", MAX_CONSECUTIVE_VICTORIES)	

#Destroy the neural network clone
if returnValue==0:
	returnValue = tRex.destroyNeuralNetwork(byref(neuralNetworkClone))

#Save the best neural network
if returnValue==0 and not neuralNetworkFileFound:
	returnValue = tRex.saveNeuralNetwork(NEURAL_NETWORK_FILE_PATH, bestNeuralNetwork)

reward = 0;

print("----- FINAL GAME -----")

env = gym.make("ALE/MsPacman-v5", repeat_action_probability=0, render_mode="human", obs_type="ram")

observation, info = env.reset()

while returnValue==0:
	c_ubyte_p = POINTER(c_ubyte)
	myOutputArray = c_ubyte_p()
	myOutputArraySize = c_int()
		
	#print(observation)

	returnValue = computeNeuralNetworkInput(observation, bestNeuralNetwork)

	if returnValue==0:
		returnValue = tRex.computeNeuralNetworkOutput(bestNeuralNetwork, byref(myOutputArray), byref(myOutputArraySize))

	if returnValue==0 and myOutputArraySize.value!=NEURAL_NETWORK_NUMBER_OF_OUTPUTS:
		returnValue = NEURAL_NETWORK_ERROR_CODE

	if returnValue==0:
		action = parseNeuralNetworkOutput(myOutputArray, myOutputArraySize.value)
		
	observation, newReward, terminated, truncated, info =  env.step(action)

	reward += newReward

	print("Reward:", reward)
	print("Action:", action)

	if terminated or truncated:
		break

env.close()

#Destroy the best neural network
if returnValue==0:
	returnValue = tRex.destroyNeuralNetwork(byref(bestNeuralNetwork))

if returnValue!=0:
	print("ERROR", returnValue)


# This code can be used to get the observation space and action space of other games
"""
print(env.action_space)
#> Discrete(9)
print(env.observation_space)
#> Box(0, 255, (128,), uint8)
print(env.action_space.high)
#> [1.]
print(env.action_space.low)
#> [-1.]
print(env.observation_space.high)
#> [255 ... 255] (128)
print(env.observation_space.low)
#> [0 ... 0] (128)
sys.exit()
"""
