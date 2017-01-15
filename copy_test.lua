require 'nn'
require 'gnuplot'
local ntm = require 'ntm'

-- Parameters
local INPUT_SIZE = 3
local OUTPUT_SIZE = 3
local MEMORY_SLOTS = 10 -- Number of addressable slots in memory
local MEMORY_SIZE = 3 -- Size of each memory slot
local CONTROLLER_SIZE = 5
local SHIFT_SIZE = 1

local TEMPORAL_HORIZON = 10

-- Generate random data to copy
local INPUT_LENGTH = 1000
local input = torch.rand(INPUT_LENGTH, INPUT_SIZE)

-- Other inputs of the NTM are zeros
local dataRead = torch.Tensor(MEMORY_SIZE):zero()
local memory = torch.Tensor(MEMORY_SLOTS, MEMORY_SIZE):zero()
local readWeights = torch.Tensor(MEMORY_SLOTS):zero()
local writeWeights = torch.Tensor(MEMORY_SLOTS):zero()

-- Create the model
local model = ntm.NTM(INPUT_SIZE, OUTPUT_SIZE, MEMORY_SLOTS, MEMORY_SIZE, CONTROLLER_SIZE, SHIFT_SIZE, TEMPORAL_HORIZON)

-- Create the criterion
criterion = nn.ParallelCriterion()
for t = 1, TEMPORAL_HORIZON do
    criterion:add(nn.MSECriterion())
end

-- Gradient descent

-- Hyper-paramètres
local learning_rate = 0.01
local max_epoch = 200
local all_losses = {}

for iteration = 1, max_epoch do
    model:zeroGradParameters()

    -- Select slice (of size TEMPORAL_HORIZON) of the global input as a training sample
    local idx = math.random(1, INPUT_LENGTH - TEMPORAL_HORIZON + 1)

    local sample = input[{{idx, idx + TEMPORAL_HORIZON - 1}}]
    modelInput = ntm.prepareModelInput(sample, dataRead, memory, readWeights, writeWeights)

    -- Forward
    modelOutput = model:forward(modelInput)
    output = ntm.unpackModelOutput(modelOutput, TEMPORAL_HORIZON)

    -- Target output is the same as input
    local loss = criterion:forward(output, sample) / (TEMPORAL_HORIZON * INPUT_SIZE)

    -- Backward
    local delta = criterion:backward(output, sample)
    -- Prepare delta essentially like the input, but with zeros for delta on everything except the model's output
    modelDelta = ntm.prepareModelInput(delta, dataRead, memory, readWeights, writeWeights)
    model:backward(modelInput, modelDelta)
    model:updateParameters(learning_rate)

    print('After epoch ' .. iteration .. ', loss is ' .. loss)
    all_losses[iteration] = loss
end
gnuplot.plot(torch.Tensor(all_losses))
