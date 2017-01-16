require 'nn'
require 'gnuplot'
local ntm = require 'ntm'

-- Parameters
local INPUT_SIZE = 5 + 1
local OUTPUT_SIZE = 5 + 1
local MEMORY_SLOTS = 128 -- Number of addressable slots in memory
local MEMORY_SIZE = 20 -- Size of each memory slot
local CONTROLLER_SIZE = 100
local SHIFT_SIZE = 1

local COPY_SIZE = 20
local TEMPORAL_HORIZON = 2 * COPY_SIZE + 1

-- Other inputs of the NTM are zeros
local dataRead = torch.Tensor(MEMORY_SIZE):zero()
local memory = torch.Tensor(MEMORY_SLOTS, MEMORY_SIZE):zero()
local readWeights = torch.Tensor(MEMORY_SLOTS):zero()
local writeWeights = torch.Tensor(MEMORY_SLOTS):zero()

-- Create the model
local model = ntm.NTM(INPUT_SIZE, OUTPUT_SIZE, MEMORY_SLOTS, MEMORY_SIZE, CONTROLLER_SIZE, SHIFT_SIZE, TEMPORAL_HORIZON)

-- Create the criterions
local criterion = nn.BCECriterion()

-- Gradient descent

-- Hyper-paramètres
local learning_rate = 0.01
local max_epoch = 20
local minibatch_size = 10
local all_losses = {}

local input = torch.Tensor(TEMPORAL_HORIZON, INPUT_SIZE):zero()
local delta = torch.Tensor(TEMPORAL_HORIZON, INPUT_SIZE):zero()
for iteration = 1, max_epoch do
    model:zeroGradParameters()

    local loss = 0
    for k = 1, minibatch_size do
        -- Generate random data to copy
        input:zero()
        local input_length = math.random(1, COPY_SIZE)
        input[{{1, input_length}, {2, INPUT_SIZE}}]:random(0, 1)
        input[input_length+1][1] = 1

        local modelInput = ntm.prepareModelInput(input, dataRead, memory, readWeights, writeWeights)

        -- Forward
        local modelOutput = model:forward(modelInput)
        local output = ntm.unpackModelOutput(modelOutput, TEMPORAL_HORIZON)

        -- Target output is the same as input
        output = output[{{input_length + 2, 2 * input_length + 1}}]
        local target = input[{{1, input_length}}]

        loss = loss + criterion:forward(output, target) / minibatch_size

        -- -- Backward
        delta:zero()
        delta[{{input_length + 2, 2 * input_length + 1}}] = criterion:backward(output, target) / minibatch_size

        -- Prepare delta essentially like the input, but with zeros for delta on everything except the model's output
        local modelDelta = ntm.prepareModelInput(delta, dataRead, memory, readWeights, writeWeights)
        model:backward(modelInput, modelDelta)

        if iteration == max_epoch and k == minibatch_size then
            print(target)
            print(output)
        end
    end
    model:updateParameters(learning_rate)

    print('After epoch ' .. iteration .. ', loss is ' .. loss)
    all_losses[iteration] = loss
end
gnuplot.plot(torch.Tensor(all_losses))
