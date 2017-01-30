require 'nn'
require 'optim'
require 'gnuplot'
ntm = require '../models/ntm'
require '../models/rmsprop'


--[[

This NTM is trained to copy an array of random bits.
The size of the array is chosen at random each time, and we add an end delimiter on a separate channel (here, column).

0  1  0  0  0  0   tensor to copy
0  1  1  0  1  0   tensor to copy
0  1  0  1  1  1   tensor to copy
1  0  0  0  0  0   end deliminter, column 1

]]--

function file_exists(filename)
   local f = io.open(filename, 'r')
   if f ~= nil then
       io.close(f)
       return true
   else
       return false
   end
end

-- Parameters
local INPUT_SIZE = 1 + 1
local OUTPUT_SIZE = INPUT_SIZE - 1
local MEMORY_SLOTS = 20 -- Number of addressable slots in memory
local MEMORY_SIZE = 4 -- Size of each memory slot
local CONTROLLER_SIZE = 20
local SHIFT_SIZE = 1

local COPY_SIZE = 5
local TEMPORAL_HORIZON = 1 + 2 * COPY_SIZE

-- Other inputs of the NTM are zeros
local dataRead = torch.Tensor(MEMORY_SIZE):zero()
local memory = torch.Tensor(MEMORY_SLOTS, MEMORY_SIZE):zero()
-- local readWeights = torch.Tensor(MEMORY_SLOTS):zero()
-- local writeWeights = torch.Tensor(MEMORY_SLOTS):zero()

-- Initialize read/write weights to a vector that stimulates
-- reading/writing at slot 1 first
local readWeights = torch.Tensor(MEMORY_SLOTS):zero()
readWeights[1] = 1.0
local writeWeights = readWeights:clone()

-- Create the criterions
local criterion = nn.BCECriterion()

local train = true
local saveFilename = 'saves/copy.mdl'

-- Create the model
models = ntm.NTM(INPUT_SIZE, OUTPUT_SIZE, MEMORY_SLOTS, MEMORY_SIZE, CONTROLLER_SIZE, SHIFT_SIZE, TEMPORAL_HORIZON)
if file_exists(saveFilename) then
    print('Loaded model from ' .. saveFilename)
    temp = torch.load(saveFilename)
    local tempParams, _ = temp:getParameters()
    local modelParams, _ = models[TEMPORAL_HORIZON]:getParameters()
    modelParams:copy(tempParams)
end

if train then
    -- Gradient descent

    -- Hyper-paramètres
    local config = {
        learningRate = 1e-3,
        momentum = 0.9,
        decay = 0.95,
        alpha = 0.9 -- decay for optim.rmsprop
    }
    local allLosses = {}
    local incremental = true
    local iterationsPerIncrement = 4000

    local maxLength = incremental and 1 or COPY_SIZE
    local maxEpoch = incremental and (iterationsPerIncrement * COPY_SIZE * (COPY_SIZE + 1) / 2) or 60000

    local params, gradParams = models[TEMPORAL_HORIZON]:getParameters()
    local input = torch.Tensor(TEMPORAL_HORIZON, INPUT_SIZE):zero()
    local delta = torch.Tensor(TEMPORAL_HORIZON, OUTPUT_SIZE):zero()
    local target = torch.Tensor(TEMPORAL_HORIZON, OUTPUT_SIZE):zero()
    local output

    local avgLoss = 0.0
    local avgMinLoss = 0.0
    local avgMaxLoss = 0.0
    local avgDecay = 1.0 / 25
    local allAvgLoss = {}
    local allAvgMinLoss = {}
    local allAvgMaxLoss = {}

    lastChange = 0
    for iteration = 1, maxEpoch do
        if incremental and ((iteration - lastChange) % (maxLength * iterationsPerIncrement) == 0) then
            lastChange = iteration
            maxLength = math.min(COPY_SIZE, maxLength + 1)
            print('Change at ' .. iteration .. ' to ' .. maxLength)
        end

        function feval(params)
            gradParams:zero()

            -- Generate random data to copy
            input:zero()
            local input_length = math.random(1, maxLength)
            currentModelSize = 1 + 2 * input_length
            model = models[currentModelSize]
            input[{{1, input_length}, {2, INPUT_SIZE}}]:random(0, 1)
            input[input_length + 1][1] = 1 -- end delimiter

            local modelInput = ntm.prepareModelInput(input[{{1, currentModelSize}}], dataRead, memory, readWeights, writeWeights)

            -- Forward
            local modelOutput = model:forward(modelInput)
            output = ntm.unpackModelOutput(modelOutput, currentModelSize)

            -- Target output is the same as input
            target:zero()
            target[{{input_length + 2, 1 + 2 * input_length}}] = input[{{1, input_length}, {2, INPUT_SIZE}}]

            -- output:clamp(1e-10, 1.0 - 1e-10) -- Clamp the output to prevent huge BCE loss
            local loss = criterion:forward(output, target[{{1, currentModelSize}}])

            -- -- Backward
            delta = criterion:backward(output, target[{{1, currentModelSize}}])
            -- delta:mul(input_length)

            -- Prepare delta essentially like the input, but with zeros for delta on everything except the model's output
            local modelDelta = ntm.prepareModelInput(delta, dataRead, memory, readWeights, writeWeights)
            model:backward(modelInput, modelDelta)

            return loss, gradParams
        end

        local _, loss = optim.rmsprop(feval, params, config)
        loss = loss[1]
        avgLoss = avgDecay * loss + (1.0 - avgDecay) * avgLoss
        avgMinLoss = math.min(loss, avgDecay * loss + (1.0 - avgDecay) * avgMinLoss)
        avgMaxLoss = math.max(loss, avgDecay * loss + (1.0 - avgDecay) * avgMaxLoss)

        if iteration % 25 == 0 then
            print('After epoch ' .. iteration .. ', loss is ' .. loss)
            print('Average loss: ' .. avgLoss)
            print('Learning rate: ' .. config.learningRate)
            print('Grad: ' .. gradParams:min() .. ' ' .. gradParams:max())
            print('Output:')
            print(output)
            print('Target:')
            print(target[{{1, currentModelSize}}])
            print('')

            allLosses[iteration/25] = loss
            allAvgLoss[iteration/25] = avgLoss
            allAvgMinLoss[iteration/25] = avgMinLoss
            allAvgMaxLoss[iteration/25] = avgMaxLoss
        end

        if iteration % 10000 == 0 then
            torch.save(saveFilename .. '.' .. iteration, models[TEMPORAL_HORIZON])
            print('Saved checkpoint to ' .. saveFilename .. '.' .. iteration)
        end
    end

    allLosses = torch.Tensor(allLosses):log()
    allAvgLoss = torch.Tensor(allAvgLoss):log()
    allAvgMinLoss = torch.Tensor(allAvgMinLoss):log()
    allAvgMaxLoss = torch.Tensor(allAvgMaxLoss):log()

    x = torch.range(1, maxEpoch, 25)
    yy = torch.cat(x, allAvgMinLoss, 2)
    yy = torch.cat(yy, allAvgMaxLoss, 2)
    gnuplot.plot({yy, 'with filledcurves fill transparent solid 0.5'}, {x, allAvgMaxLoss, 'with lines ls 1'}, {x, allAvgMinLoss, 'with lines ls 1'}, {x, allAvgLoss, 'with lines ls 1'})

    torch.save(saveFilename, models[TEMPORAL_HORIZON])
    print('Saved model to ' .. saveFilename)
end

-- Evaluation

local numEvaluations = 200
local input = torch.Tensor(TEMPORAL_HORIZON, INPUT_SIZE):zero()
local target = torch.Tensor(TEMPORAL_HORIZON, OUTPUT_SIZE):zero()
local output
local sumLoss = 0.0
local sumSqrLoss = 0.0

for iteration = 1, numEvaluations do
    -- Generate random data to copy
    input:zero()
    local input_length = math.random(1, COPY_SIZE)
    local currentModelSize = 1 + 2 * input_length
    model = models[currentModelSize]
    input[{{1, input_length}, {2, INPUT_SIZE}}]:random(0, 1)
    input[input_length + 1][1] = 1 -- end delimiter

    local modelInput = ntm.prepareModelInput(input[{{1, currentModelSize}}], dataRead, memory, readWeights, writeWeights)

    -- Forward
    local modelOutput = model:forward(modelInput)
    output = ntm.unpackModelOutput(modelOutput, currentModelSize)

    -- Target output is the same as input
    target:zero()
    target[{{input_length + 2, 1 + 2 * input_length}}] = input[{{1, input_length}, {2, INPUT_SIZE}}]

    local loss = criterion:forward(output, target[{{1, currentModelSize}}])
    sumLoss = sumLoss + loss
    sumSqrLoss = sumSqrLoss + loss*loss
end

avgLoss = sumLoss / numEvaluations
avgSqrLoss = sumSqrLoss / numEvaluations
stdDevLoss = math.sqrt(avgSqrLoss - avgLoss*avgLoss)

print('Loss: ' .. avgLoss .. ' +- ' .. stdDevLoss)
