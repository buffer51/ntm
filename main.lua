require 'nn'
local ntm = require 'ntm'

-- Parameters
local INPUT_SIZE = 4
local OUTPUT_SIZE = 2
local MEMORY_SLOTS = 10 -- Number of addressable slots in memory
local MEMORY_SIZE = 3 -- Size of each memory slot
local CONTROLLER_SIZE = 5
local SHIFT_SIZE = 1

-- Test the NTM module
print('----------------------------')
print('-----------Inputs-----------')
print('----------------------------')
print('')

local testInput = torch.Tensor(INPUT_SIZE):zero()
testInput[1] = 1.0
testInput[3] = 2.0
print('Input:')
print(testInput)

local testMemoryP = torch.Tensor(MEMORY_SLOTS, MEMORY_SIZE):zero()
for i = 1, MEMORY_SLOTS do
    testMemoryP[i]:fill(i)
end
print('Previous Memory:')
print(testMemoryP)

local testReadWeightsP = torch.Tensor(MEMORY_SLOTS):zero()
testReadWeightsP[2] = 0.5
testReadWeightsP[3] = 0.3
testReadWeightsP[8] = 0.2
print('Previous Read Weights:')
print(testReadWeightsP)

local testDataReadP = torch.Tensor(MEMORY_SIZE):zero()
testDataReadP:addmv(testMemoryP:t(), testReadWeightsP)
print('Previous Data Read:')
print(testDataReadP)

local testWriteWeightsP = torch.Tensor(MEMORY_SLOTS):zero()
testWriteWeightsP[1] = 0.1
testWriteWeightsP[5] = 0.8
testWriteWeightsP[10] = 0.1
print('Previous Write Weights:')
print(testWriteWeightsP)

local module = ntm.NTM(INPUT_SIZE, OUTPUT_SIZE, MEMORY_SLOTS, MEMORY_SIZE, CONTROLLER_SIZE, SHIFT_SIZE)

-- Forward pass
print('----------------------------')
print('-----------Output-----------')
print('----------------------------')
print('')

local out = module:forward{testInput, testDataReadP, testMemoryP, testReadWeightsP, testWriteWeightsP}

local testOutput = out[1]
print('Output:')
print(testOutput)

local testDataRead = out[2]
print('Data Read:')
print(testDataRead)

local testMemory = out[3]
print('Memory:')
print(testMemory)

local testReadWeights = out[4]
print('Read Weights:')
print(testReadWeights)

local testWriteWeights = out[5]
print('Write Weights:')
print(testWriteWeights)
