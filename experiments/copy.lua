require 'nn'
require 'optim'
require 'gnuplot'
ntm = require '../models/ntm'


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

-- Task parameters
local COPY_SIZE = 1 -- Width of copied sequences
local COPY_LENGTH = 5 -- Maximum length of copied sequences

-- Control variables
local train = true -- Set to true for training, false for evaluation only
local loadFilename = 'saves/to_load.mdl' -- If not nil, will be loaded (for evaluation or further training)
local saveFilename = 'saves/copy.mdl' -- Filename of saves

-- NTM Parameters
local INPUT_SIZE = 1 + COPY_SIZE
local OUTPUT_SIZE = COPY_SIZE
local MEMORY_SLOTS = 20 -- Number of addressable slots in memory
local MEMORY_SIZE = 4 -- Size of each memory slot
local CONTROLLER_SIZE = 20
local SHIFT_SIZE = 1
local TEMPORAL_HORIZON = 1 + 2 * COPY_LENGTH

-- Initialization
math.randomseed(os.time())

local dataRead = torch.Tensor(MEMORY_SIZE):zero()
local memory = torch.Tensor(MEMORY_SLOTS, MEMORY_SIZE):zero()
-- Initialize read/write weights to a vector that stimulates
-- reading/writing at slot 1 first
local readWeights = torch.Tensor(MEMORY_SLOTS):zero()
readWeights[1] = 1.0
local writeWeights = readWeights:clone()

-- Create the criterion
local criterion = nn.BCECriterion()

-- Create the models
models = ntm.NTM(INPUT_SIZE, OUTPUT_SIZE, MEMORY_SLOTS, MEMORY_SIZE, CONTROLLER_SIZE, SHIFT_SIZE, TEMPORAL_HORIZON)

-- Load existing model, if set
if loadFilename and file_exists(loadFilename) then
    print('Loaded model from ' .. loadFilename)
    temp = torch.load(loadFilename)
    local tempParams, _ = temp:getParameters()
    local modelParams, _ = models[TEMPORAL_HORIZON]:getParameters()
    modelParams:copy(tempParams)
end

if train then
    -- Gradient descent

    -- Hyper parameters
    local config = {
        learningRate = 1e-3,
        alpha = 0.9
    }

    -- Type of training to use
    local incremental = true -- If true, will train on sequences of increasing length

    local maxEpoch, maxLength
    if incremental then
        iterationsPerIncrement = 4000
        lastChange = 0

        -- Start with sequences of length 1, and slowly increase
        maxLength = 1
        maxEpoch = iterationsPerIncrement * COPY_LENGTH * (COPY_LENGTH + 1) / 2
    else
        maxLength = COPY_LENGTH
        maxEpoch = 60000
    end

    local params, gradParams = models[TEMPORAL_HORIZON]:getParameters()
    local input = torch.Tensor(TEMPORAL_HORIZON, INPUT_SIZE):zero()
    local delta = torch.Tensor(TEMPORAL_HORIZON, OUTPUT_SIZE):zero()
    local target = torch.Tensor(TEMPORAL_HORIZON, OUTPUT_SIZE):zero()
    local output

    -- Reporting
    local reportingStep = 50
    local avgLoss = 0.0 -- Running average of loss
    local avgMinLoss = 0.0
    local avgMaxLoss = 0.0
    local avgDecay = 1.0 / reportingStep
    local allLosses = {}
    local allAvgLosses = {}
    local allAvgMinLosses = {}
    local allAvgMaxLosses = {}

    for iteration = 1, maxEpoch do
        if incremental and ((iteration - lastChange) % (maxLength * iterationsPerIncrement) == 0) then
            lastChange = iteration
            maxLength = math.min(COPY_LENGTH, maxLength + 1)
        end

        function feval(params)
            gradParams:zero()

            -- Generate random data to copy
            input:zero()
            local inputLength = math.random(1, maxLength)
            currentModelSize = 1 + 2 * inputLength
            model = models[currentModelSize]
            input[{{1, inputLength}, {2, INPUT_SIZE}}]:random(0, 1)
            input[inputLength + 1][1] = 1 -- end delimiter

            local modelInput = ntm.prepareModelInput(input[{{1, currentModelSize}}], dataRead, memory, readWeights, writeWeights)

            -- Forward
            local modelOutput = model:forward(modelInput)
            output = ntm.unpackModelOutput(modelOutput, currentModelSize)

            -- Target output is the same as input
            target:zero()
            target[{{inputLength + 2, 1 + 2 * inputLength}}] = input[{{1, inputLength}, {2, INPUT_SIZE}}]

            local loss = criterion:forward(output, target[{{1, currentModelSize}}])

            -- -- Backward
            delta = criterion:backward(output, target[{{1, currentModelSize}}])

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

        if iteration % reportingStep == 0 then
            print('After epoch ' .. iteration .. ', loss is ' .. loss)
            print('Average loss: ' .. avgLoss)
            print('Learning rate: ' .. config.learningRate)
            print('Grad: ' .. gradParams:min() .. ' ' .. gradParams:max())
            print('Output:')
            print(output)
            print('Target:')
            print(target[{{1, currentModelSize}}])
            print('')

            allLosses[iteration/reportingStep] = loss
            allAvgLosses[iteration/reportingStep] = avgLoss
            allAvgMinLosses[iteration/reportingStep] = avgMinLoss
            allAvgMaxLosses[iteration/reportingStep] = avgMaxLoss
        end

        if iteration % 10000 == 0 then
            torch.save(saveFilename .. '.' .. iteration, models[TEMPORAL_HORIZON])
            print('Saved checkpoint to ' .. saveFilename .. '.' .. iteration)
        end
    end

    allLosses = torch.Tensor(allLosses):log()
    allAvgLosses = torch.Tensor(allAvgLosses):log()
    allAvgMinLosses = torch.Tensor(allAvgMinLosses):log()
    allAvgMaxLosses = torch.Tensor(allAvgMaxLosses):log()

    x = torch.range(reportingStep, maxEpoch, reportingStep)
    yy = torch.cat(x, allAvgMinLosses, 2)
    yy = torch.cat(yy, allAvgMaxLosses, 2)
    gnuplot.plot({yy, 'with filledcurves fill transparent solid 0.5'}, {x, allAvgMaxLosses, 'with lines ls 1'}, {x, allAvgMinLosses, 'with lines ls 1'}, {x, allAvgLosses, 'with lines ls 1'})

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
    local inputLength = math.random(1, COPY_LENGTH)
    local currentModelSize = 1 + 2 * inputLength
    model = models[currentModelSize]
    input[{{1, inputLength}, {2, INPUT_SIZE}}]:random(0, 1)
    input[inputLength + 1][1] = 1 -- end delimiter

    local modelInput = ntm.prepareModelInput(input[{{1, currentModelSize}}], dataRead, memory, readWeights, writeWeights)

    -- Forward
    local modelOutput = model:forward(modelInput)
    output = ntm.unpackModelOutput(modelOutput, currentModelSize)

    -- Target output is the same as input
    target:zero()
    target[{{inputLength + 2, 1 + 2 * inputLength}}] = input[{{1, inputLength}, {2, INPUT_SIZE}}]

    local loss = criterion:forward(output, target[{{1, currentModelSize}}])
    sumLoss = sumLoss + loss
    sumSqrLoss = sumSqrLoss + loss*loss
end

avgLoss = sumLoss / numEvaluations
avgSqrLoss = sumSqrLoss / numEvaluations
stdDevLoss = math.sqrt(avgSqrLoss - avgLoss*avgLoss)

print('Loss: ' .. avgLoss .. ' +- ' .. stdDevLoss)
