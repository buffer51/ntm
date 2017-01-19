require 'nn'
require 'gnuplot'
ntm = require 'ntm'
require 'rmsprop'

--[[

The aim of this NTM is to repeat a tensor a number of times, like in a for loop.
We give a tensor of random(0,1) as input, with a int on a specific channel (column) to give the number of copies.
Then, we give an end delimiter on another specific column.

Example of input:

3  0  0  0  0  0  0   Number of copies to make, column 1
0  0  1  0  0  0  0   tensor to copy
0  0  1  1  0  1  0   tensor to copy
0  0  1  0  1  1  1   tensor to copy
0  1  0  0  0  0  0   end deliminter, column 2


]]--



-- Parameters
local INPUT_SIZE = 5 + 2
local OUTPUT_SIZE = 5 + 2
local MEMORY_SLOTS = 128 -- Number of addressable slots in memory
local MEMORY_SIZE = 20 -- Size of each memory slot
local CONTROLLER_SIZE = 100
local SHIFT_SIZE = 1
local MAX_NUMBER_0F_COPY = 5

local COPY_SIZE = 40
local TEMPORAL_HORIZON = 2 + 2 * COPY_SIZE

-- Other inputs of the NTM are zeros
local dataRead = torch.Tensor(MEMORY_SIZE):zero()
local memory = torch.Tensor(MEMORY_SLOTS, MEMORY_SIZE):zero()

-- Initialize read/write weights to a vector that stimulates
-- reading/writing at slot 1 first
local readWeights = nn.SoftMax():forward(torch.range(MEMORY_SLOTS, 1, -1))
local writeWeights = readWeights:clone()

-- Create the model
local model = ntm.NTM(INPUT_SIZE, OUTPUT_SIZE, MEMORY_SLOTS, MEMORY_SIZE, CONTROLLER_SIZE, SHIFT_SIZE, TEMPORAL_HORIZON)

-- Create the criterions
local criterion = nn.BCECriterion()

-- Meta parameters
local config = {
    learningRate = 1e-4,
    momentum = 0.9,
    decay = 0.95
}
local maxEpoch = 5000
local allLosses = {}

local params, gradParams = model:getParameters()
local input = torch.Tensor(TEMPORAL_HORIZON, INPUT_SIZE):zero()
local delta = torch.Tensor(TEMPORAL_HORIZON, INPUT_SIZE):zero()
local output, target, numberOfCopy

for iteration = 1, maxEpoch do
    function feval(params)
        gradParams:zero()

        -- Generate random data to copy
        input:zero()

        numberOfCopy = math.random(1, MAX_NUMBER_0F_COPY)
        local inputLength = math.random(1, COPY_SIZE / MAX_NUMBER_0F_COPY)

        input[{{2, inputLength + 1}, {3, INPUT_SIZE}}]:random(0, 1)
        input[1][1] = numberOfCopy -- number of copies
        input[inputLength + 2][2] = 1 -- end delimiter

        local modelInput = ntm.prepareModelInput(input, dataRead, memory, readWeights, writeWeights)

        -- Forward
        local modelOutput = model:forward(modelInput)
        output = ntm.unpackModelOutput(modelOutput, TEMPORAL_HORIZON)

        output = output[{{2 + inputLength + 2, 3 + inputLength * (numberOfCopy + 1)}}]
        cell = input[{{2, inputLength + 1}}]

        target = torch.repeatTensor(cell, numberOfCopy, 1)

        local loss = criterion:forward(output, target)

        -- -- Backward
        delta:zero()
        delta[{{2 + inputLength + 2, 3 + inputLength * (numberOfCopy + 1)}}] = criterion:backward(output, target)
        delta:mul(inputLength * numberOfCopy)

        -- Prepare delta essentially like the input, but with zeros for delta on everything except the model's output
        local modelDelta = ntm.prepareModelInput(delta, dataRead, memory, readWeights, writeWeights)
        model:backward(modelInput, modelDelta)

        return loss, gradParams
    end

    local _, loss = ntm.rmsprop(feval, params, config)
    loss = loss[1]

    if iteration % 25 == 0 then
        print('After epoch ' .. iteration .. ', loss is ' .. loss)
        print('Grad: ' .. gradParams:min() .. ' ' .. gradParams:max())
        print('Number of copies: ' .. numberOfCopy)
        print('Output:')
        print(output)
        print('Target:')
        print(target)
        print('')
    end
    allLosses[iteration] = loss
end
gnuplot.plot(torch.Tensor(allLosses))
