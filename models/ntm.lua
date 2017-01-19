require 'nn'
require 'nngraph'
require '../modules/ScalarMulTable'
require '../modules/SmoothCosineSimilarity'
require '../modules/CircularConvolution'
require '../modules/PowTable'
require '../modules/OuterProd'

local model_utils = require '../models/model_utils'

local function NTMCell(input_size, output_size, memory_slots, memory_size, controller_size, shift_size)
    -- Inputs
    local input = nn.Identity()()
    input:annotate{
        name = 'input', description = 'Input',
        graphAttributes = {style = 'filled', fillcolor = 'green'}
    }
    local dataReadP = nn.Identity()()
    dataReadP:annotate{
        name = 'dataReadP', description = 'Previous Data Read',
        graphAttributes = {style = 'filled', fillcolor = 'green'}
    }
    local memoryP = nn.Identity()()
    memoryP:annotate{
        name = 'memoryP', description = 'Previous Memory',
        graphAttributes = {style = 'filled', fillcolor = 'green'}
    }
    local readWeightsP = nn.Identity()()
    readWeightsP:annotate{
        name = 'readWeightsP', description = 'Previous Read Weights',
        graphAttributes = {style = 'filled', fillcolor = 'green'}
    }
    local writeWeightsP = nn.Identity()()
    writeWeightsP:annotate{
        name = 'writeWeightsP', description = 'Previous Write Weights',
        graphAttributes = {style = 'filled', fillcolor = 'green'}
    }

    -- Controller
    local controllerOutput = nn.Tanh()(nn.CAddTable(){
        nn.Linear(input_size, controller_size)(input),
        nn.Linear(memory_size, controller_size)(dataReadP)
    })
    controllerOutput:annotate{
        name = 'controllerOutput', description = 'Controller Output',
        graphAttributes = {style = 'filled', fillcolor = 'orange'}
    }

    -- Output
    local output = nn.Sigmoid()(nn.Linear(controller_size, output_size)(controllerOutput))
    output:annotate{
        name = 'output', description = 'Output',
        graphAttributes = {style = 'filled', fillcolor = 'red'}
    }

    -- Memory operations
    function newWeights(memoryP, controllerOutput, weightsP)
        local k = nn.Tanh()(nn.Linear(controller_size, memory_size)(controllerOutput))
        local beta = nn.SoftPlus()(nn.Linear(controller_size, 1)(controllerOutput))
        local contentWeights = nn.SoftMax()(nn.ScalarMulTable(){nn.SmoothCosineSimilarity(){memoryP, k}, beta})

        local g = nn.Sigmoid()(nn.Linear(controller_size, 1)(controllerOutput))
        local locationWeights = nn.CAddTable(){
            nn.ScalarMulTable(){contentWeights, g},
            nn.ScalarMulTable(){weightsP, nn.AddConstant(1)(nn.MulConstant(-1)(g))}
        }

        local s = nn.SoftMax()(nn.Linear(controller_size, 2 * shift_size + 1)(controllerOutput))
        local shiftedWeights = nn.CircularConvolution(){locationWeights, s}

        local gamma = nn.AddConstant(1)(nn.SoftPlus()(nn.Linear(controller_size, 1)(controllerOutput)))
        local sharpenedWeights = nn.Normalize(1)(nn.PowTable(){shiftedWeights, gamma})

        return sharpenedWeights
    end

    -- Read
    local readWeights = newWeights(memoryP, controllerOutput, readWeightsP)
    readWeights:annotate{
        name = 'readWeights', description = 'Read Weights',
        graphAttributes = {style = 'filled', fillcolor = 'red'}
    }

    local dataRead = nn.MixtureTable(){readWeights, memoryP}
    dataRead:annotate{
        name = 'dataRead', description = 'Data Read',
        graphAttributes = {style = 'filled', fillcolor = 'red'}
    }

    -- Write
    local writeWeights = newWeights(memoryP, controllerOutput, writeWeightsP)
    writeWeights:annotate{
        name = 'writeWeights', description = 'Write Weights',
        graphAttributes = {style = 'filled', fillcolor = 'red'}
    }

    local erase = nn.Sigmoid()(nn.Linear(controller_size, memory_size)(controllerOutput))
    erase:annotate{
        name = 'erase', description = 'Erase',
        graphAttributes = {style = 'filled', fillcolor = 'orange'}
    }
    local append = nn.Tanh()(nn.Linear(controller_size, memory_size)(controllerOutput))
    append:annotate{
        name = 'append', description = 'Append',
        graphAttributes = {style = 'filled', fillcolor = 'orange'}
    }

    local memoryErased = nn.CMulTable(){memoryP, nn.AddConstant(1)(nn.MulConstant(-1)(nn.OuterProd(){writeWeights, erase}))}
    local memory = nn.CAddTable(){memoryErased, nn.OuterProd(){writeWeights, append}}
    memory:annotate{
        name = 'memory', description = 'Memory',
        graphAttributes = {style = 'filled', fillcolor = 'red'}
    }

    -- Module creation
    return nn.gModule({input, dataReadP, memoryP, readWeightsP, writeWeightsP}, {output, dataRead, memory, readWeights, writeWeights})
end


local function concatenateTables(tables)
    local result = {}

    local index = 1
    for k = 1, table.getn(tables) do
        for i = 1, table.getn(tables[k]) do
            result[index] = tables[k][i]
            index = index + 1
        end
    end

    return result
end

local function prepareModelInput(input, dataRead, memory, readWeights, writeWeights)
    local modelInput = {}
    for i = 1, input:size(1) do
        modelInput[i] = input[i]
    end

    local index = table.getn(modelInput)
    modelInput[index + 1] = dataRead
    modelInput[index + 2] = memory
    modelInput[index + 3] = readWeights
    modelInput[index + 4] = writeWeights

    return modelInput
end

local function unpackModelOutput(modelOutput, temporal_horizon)
    local output = torch.Tensor(temporal_horizon, modelOutput[1]:size(1)):zero()

    for i = 1, temporal_horizon do
        output[i] = modelOutput[i]
    end

    local dataRead = modelOutput[temporal_horizon + 1]
    local memory = modelOutput[temporal_horizon + 2]
    local readWeights = modelOutput[temporal_horizon + 3]
    local writeWeights = modelOutput[temporal_horizon + 4]

    return output, dataRead, memory, readWeights, writeWeights
end

local function NTM(input_size, output_size, memory_slots, memory_size, controller_size, shift_size, temporal_horizon)
    -- Create a recurrent model with temporal_horizon NTMs
    local module = NTMCell(input_size, output_size, memory_slots, memory_size, controller_size, shift_size)
    local ntms = model_utils.clone_many_times(module, temporal_horizon)

    -- Assemble it with nngraph
    local inputs = {}
    local outputs = {}

    local dataReads = {nn.Identity()()}
    dataReads[1]:annotate{
        name = 'dataReads[1]', description = 'First Data Read',
        graphAttributes = {style = 'filled', fillcolor = 'green'}
    }
    local memorys = {nn.Identity()()}
    memorys[1]:annotate{
        name = 'memorys[1]', description = 'First Memory',
        graphAttributes = {style = 'filled', fillcolor = 'green'}
    }
    local readWeightss = {nn.Identity()()}
    readWeightss[1]:annotate{
        name = 'readWeightss[1]', description = 'First Read Weights',
        graphAttributes = {style = 'filled', fillcolor = 'green'}
    }
    local writeWeightss = {nn.Identity()()}
    writeWeightss[1]:annotate{
        name = 'writeWeightss[1]', description = 'First Write Weights',
        graphAttributes = {style = 'filled', fillcolor = 'green'}
    }

    for t = 1, temporal_horizon do
        inputs[t] = nn.Identity()()
        inputs[t]:annotate{
            name = 'inputs[' .. t .. ']', description = 'Input ' .. t,
            graphAttributes = {style = 'filled', fillcolor = 'green'}
        }

        local out = ntms[t]{inputs[t], dataReads[t], memorys[t], readWeightss[t], writeWeightss[t]}
        out:annotate{
            name = 'ntms[' .. t .. ']', description = 'NTM ' .. t,
            graphAttributes = {style = 'filled', fillcolor = 'orange'}
        }

        outputs[t] = nn.SelectTable(1)(out)
        outputs[t]:annotate{
            name = 'outputs[' .. t .. ']', description = 'Output ' .. t,
            graphAttributes = {style = 'filled', fillcolor = 'red'}
        }
        dataReads[t+1] = nn.SelectTable(2)(out)
        memorys[t+1] = nn.SelectTable(3)(out)
        readWeightss[t+1] = nn.SelectTable(4)(out)
        writeWeightss[t+1] = nn.SelectTable(5)(out)
    end

    local modelInput = concatenateTables({inputs, {dataReads[1], memorys[1], readWeightss[1], writeWeightss[1]}})
    local modelOutput = concatenateTables({outputs, {dataReads[temporal_horizon+1], memorys[temporal_horizon+1], readWeightss[temporal_horizon+1], writeWeightss[temporal_horizon+1]}})

    return nn.gModule(modelInput, modelOutput)
end

-- Example with graph
local ntm = NTMCell(4, 2, 10, 3, 5, 1)
graph.dot(ntm.fg, 'NTM', 'NTM')

local model = NTM(4, 2, 10, 3, 5, 1, 10)
graph.dot(model.fg, 'Model', 'model')

return {NTMCell = NTMCell, NTM = NTM, prepareModelInput = prepareModelInput, unpackModelOutput = unpackModelOutput}
