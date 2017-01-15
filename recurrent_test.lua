require 'nn'
local ntm = require 'ntm'

-- Parameters
local INPUT_SIZE = 4
local OUTPUT_SIZE = 2
local MEMORY_SLOTS = 10 -- Number of addressable slots in memory
local MEMORY_SIZE = 3 -- Size of each memory slot
local CONTROLLER_SIZE = 5
local SHIFT_SIZE = 1

local TEMPORAL_HORIZON = 10

-- Test the recurrent architecture
local testInput = torch.Tensor(TEMPORAL_HORIZON, INPUT_SIZE):zero()
for i = 1, TEMPORAL_HORIZON do
    testInput[i][1 + ((i-1) % 4)] = i
end
print('Input:')
print(testInput)

local testDataRead = torch.Tensor(MEMORY_SIZE):zero()
local testMemory = torch.Tensor(MEMORY_SLOTS, MEMORY_SIZE):zero()
local testReadWeights = torch.Tensor(MEMORY_SLOTS):zero()
local testWriteWeights = torch.Tensor(MEMORY_SLOTS):zero()
testModelInput = ntm.prepareModelInput(testInput, testDataRead, testMemory, testReadWeights, testWriteWeights)

-- Forward pass
local model = ntm.NTM(INPUT_SIZE, OUTPUT_SIZE, MEMORY_SLOTS, MEMORY_SIZE, CONTROLLER_SIZE, SHIFT_SIZE, TEMPORAL_HORIZON)
testModelOutput = model:forward(testModelInput)
testOutput, testDataRead, testMemory, testReadWeights, testWriteWeights = ntm.unpackModelOutput(testModelOutput, TEMPORAL_HORIZON)

print('Output:')
print(testOutput)

print('Memory:')
print(testMemory)
