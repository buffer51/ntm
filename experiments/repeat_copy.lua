require 'nn'
require 'optim'
require 'gnuplot'
ntm = require '../models/ntm'


--[[

This tasks aims to train a NTM to repeat a sequence a given number of times,
thus emulating a for loop.

The input is the random sequence of 0 & 1s, followed by a marker repeated as
many times as we want copies, and a final end delimiter.

Example of input: a 3 x 5 tensor to copy 2 times

0  0  1  0  0  0  0   Tensor to copy
0  0  1  1  0  1  0   Tensor to copy
0  0  1  0  1  1  1   Tensor to copy
0  1  0  0  0  0  0   Copy marker
0  1  0  0  0  0  0   Copy marker
1  0  0  0  0  0  0   End deliminter

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
local COPY_SIZE = 4 -- Width of copied sequences
local COPY_LENGTH = 5 -- Maximum length of copied sequences
local REPEAT_TIMES = 3 -- Maximum number of repetitions

-- Control variables
local train = true -- Set to true for training, false for evaluation only
local loadFilename = 'save/to_load.mdl' -- If not nil, will be loaded (for evaluation or further training)
local saveFilename = 'saves/repeat_copy.mdl' -- Filename of saves

-- NTM Parameters
local INPUT_SIZE = 2 + COPY_SIZE
local OUTPUT_SIZE = COPY_SIZE
local MEMORY_SLOTS = 50 -- Number of addressable slots in memory
local MEMORY_SIZE = 8 -- Size of each memory slot
local CONTROLLER_SIZE = 50
local SHIFT_SIZE = 1
local TEMPORAL_HORIZON = 1 + REPEAT_TIMES + (1 + REPEAT_TIMES) * COPY_LENGTH

-- Initialization
math.randomseed(os.time())

local dataRead = torch.Tensor(MEMORY_SIZE):zero()
local memory = torch.Tensor(MEMORY_SLOTS, MEMORY_SIZE):fill(1e-6)
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

    local maxEpoch = 100000

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
        function feval(params)
            gradParams:zero()

            -- Generate random data to copy
            input:zero()
            inputLength = math.random(3, COPY_LENGTH) -- 3 or more
            repetitions = math.random(1, REPEAT_TIMES)
            currentModelSize = 1 + repetitions + (1 + repetitions) * inputLength
            model = models[currentModelSize]
            input[{{1, inputLength}, {3, INPUT_SIZE}}]:random(0, 1)
            input[{{inputLength + 1, inputLength + repetitions}, 2}] = 1 -- Repeat marker
            input[inputLength + repetitions + 1][1] = 1 -- End delimiter

            local modelInput = ntm.prepareModelInput(input[{{1, currentModelSize}}], dataRead, memory, readWeights, writeWeights)

            -- Forward
            local modelOutput = model:forward(modelInput)
            output = ntm.unpackModelOutput(modelOutput, currentModelSize)

            -- Target output is the same as input
            target:zero()
            for i = 1, repetitions do
                target[{{2 + repetitions + i * inputLength, 1 + repetitions + (1 + i) * inputLength}}] = input[{{1, inputLength}, {3, INPUT_SIZE}}]
            end

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
            print('Length: ' .. inputLength .. ', repetitions: ' .. repetitions)
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

    allLosses = torch.Tensor(allLosses):log():div(math.log(10))
    allAvgLosses = torch.Tensor(allAvgLosses):log():div(math.log(10))
    allAvgMinLosses = torch.Tensor(allAvgMinLosses):log():div(math.log(10))
    allAvgMaxLosses = torch.Tensor(allAvgMaxLosses):log():div(math.log(10))

    x = torch.range(reportingStep, maxEpoch, reportingStep)
    yy = torch.cat(x, allAvgMinLosses, 2)
    yy = torch.cat(yy, allAvgMaxLosses, 2)
    gnuplot.plot({yy, 'with filledcurves fill transparent solid 0.5'}, {x, allAvgMaxLosses, 'with lines ls 1'}, {x, allAvgMinLosses, 'with lines ls 1'}, {x, allAvgLosses, 'with lines ls 1'})

    torch.save(saveFilename, models[TEMPORAL_HORIZON])
    print('Saved model to ' .. saveFilename)
end

-- Evaluation

local numEvaluations = 3000
local input = torch.Tensor(TEMPORAL_HORIZON, INPUT_SIZE):zero()
local target = torch.Tensor(TEMPORAL_HORIZON, OUTPUT_SIZE):zero()
local output

local BCELoss = nn.BCECriterion()
local MSELoss = nn.MSECriterion()

local sumLoss = torch.Tensor(3):zero() -- BCE, MSE & hard error
local sumSqrLoss = torch.Tensor(3):zero()


for iteration = 1, numEvaluations do
    -- Generate random data to copy
    input:zero()
    inputLength = math.random(3, COPY_LENGTH) -- 3 or more
    repetitions = math.random(1, REPEAT_TIMES)
    currentModelSize = 1 + repetitions + (1 + repetitions) * inputLength
    model = models[currentModelSize]
    input[{{1, inputLength}, {3, INPUT_SIZE}}]:random(0, 1)
    input[{{inputLength + 1, inputLength + repetitions}, 2}] = 1 -- Repeat marker
    input[inputLength + repetitions + 1][1] = 1 -- End delimiter

    local modelInput = ntm.prepareModelInput(input[{{1, currentModelSize}}], dataRead, memory, readWeights, writeWeights)

    -- Forward
    local modelOutput = model:forward(modelInput)
    output = ntm.unpackModelOutput(modelOutput, currentModelSize)

    -- Target output is the same as input
    target:zero()
    for i = 1, repetitions do
        target[{{2 + repetitions + i * inputLength, 1 + repetitions + (1 + i) * inputLength}}] = input[{{1, inputLength}, {3, INPUT_SIZE}}]
    end

    local loss = BCELoss:forward(output, target[{{1, currentModelSize}}])
    sumLoss[1] = sumLoss[1] + loss
    sumSqrLoss[1] = sumSqrLoss[1] + loss*loss

    local loss = MSELoss:forward(output, target[{{1, currentModelSize}}])
    sumLoss[2] = sumLoss[2] + loss
    sumSqrLoss[2] = sumSqrLoss[2] + loss*loss

    local loss = MSELoss:forward(output:gt(0.5):double(), target[{{1, currentModelSize}}])
    sumLoss[3] = sumLoss[3] + loss
    sumSqrLoss[3] = sumSqrLoss[3] + loss*loss
end

local avgLoss = sumLoss:div(numEvaluations)
local avgSqrLoss = sumSqrLoss:div(numEvaluations)
local stdDevLoss = (avgSqrLoss - avgLoss:cmul(avgLoss)):sqrt()

print('On ' .. numEvaluations .. ' evaluations:')
print('BCE: ' .. avgLoss[1] .. ' +- ' .. stdDevLoss[1])
print('MSE: ' .. avgLoss[2] .. ' +- ' .. stdDevLoss[2])
print('Hard error: ' .. avgLoss[3] .. ' +- ' .. stdDevLoss[3])
