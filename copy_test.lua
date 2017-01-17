require 'nn'
require 'gnuplot'
ntm = require 'ntm'
require 'rmsprop'

-- Parameters
local INPUT_SIZE = 5 + 1
local OUTPUT_SIZE = 5 + 1
local MEMORY_SLOTS = 128 -- Number of addressable slots in memory
local MEMORY_SIZE = 20 -- Size of each memory slot
local CONTROLLER_SIZE = 100
local SHIFT_SIZE = 1

local COPY_SIZE = 20
local TEMPORAL_HORIZON = 1 + 2 * COPY_SIZE

-- Other inputs of the NTM are zeros
local dataRead = torch.Tensor(MEMORY_SIZE):zero()
local memory = torch.Tensor(MEMORY_SLOTS, MEMORY_SIZE):zero()
-- local readWeights = torch.Tensor(MEMORY_SLOTS):zero()
-- local writeWeights = torch.Tensor(MEMORY_SLOTS):zero()

-- Initialize read/write weights to a vector that stimulates
-- reading/writing at slot 1 first
local readWeights = nn.SoftMax():forward(torch.range(MEMORY_SLOTS, 1, -1))
local writeWeights = readWeights:clone()

-- Create the model
local model = ntm.NTM(INPUT_SIZE, OUTPUT_SIZE, MEMORY_SLOTS, MEMORY_SIZE, CONTROLLER_SIZE, SHIFT_SIZE, TEMPORAL_HORIZON)

-- Create the criterions
local criterion = nn.BCECriterion()

-- Gradient descent

-- Hyper-paramètres
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
local output, target

for iteration = 1, maxEpoch do
    function feval(params)
        gradParams:zero()

        -- Generate random data to copy
        input:zero()
        local input_length = math.random(1, COPY_SIZE)
        input[{{1, input_length}, {2, INPUT_SIZE}}]:random(0, 1)
        input[input_length + 1][1] = 1 -- end delimiter

        local modelInput = ntm.prepareModelInput(input, dataRead, memory, readWeights, writeWeights)

        -- Forward
        local modelOutput = model:forward(modelInput)
        output = ntm.unpackModelOutput(modelOutput, TEMPORAL_HORIZON)

        -- Target output is the same as input
        output = output[{{input_length + 2, 1 + 2 * input_length}}]
        target = input[{{1, input_length}}]

        local loss = criterion:forward(output, target)

        -- -- Backward
        delta:zero()
        delta[{{input_length + 2, 1 + 2 * input_length}}] = criterion:backward(output, target)
        delta:mul(input_length)

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
        print('Output:')
        print(output)
        print('Target:')
        print(target)
        print('')
    end
    allLosses[iteration] = loss
end
gnuplot.plot(torch.Tensor(allLosses))
