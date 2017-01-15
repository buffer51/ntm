require 'nn'
local ntm = require 'ntm'
local model_utils = require 'model_utils'

local function concatenateTables(tables)
    result = {}

    index = 1
    for k = 1, table.getn(tables) do
        for i = 1, table.getn(tables[k]) do
            result[index] = tables[k][i]
            index = index + 1
        end
    end

    return result
end

-- Parameters
local INPUT_SIZE = 4
local OUTPUT_SIZE = 2
local MEMORY_SLOTS = 10 -- Number of addressable slots in memory
local MEMORY_SIZE = 3 -- Size of each memory slot
local CONTROLLER_SIZE = 5
local SHIFT_SIZE = 1

local TEMPORAL_HORIZON = 10

-- Create a recurrent model with TEMPORAL_HORIZON NTMs
local module = ntm.NTM(INPUT_SIZE, OUTPUT_SIZE, MEMORY_SLOTS, MEMORY_SIZE, CONTROLLER_SIZE, SHIFT_SIZE)
local ntms = model_utils.clone_many_times(module, TEMPORAL_HORIZON)

-- Assemble it with nngraph
inputs = {}
outputs = {}

dataReads = {nn.Identity()()}
dataReads[1]:annotate{
    name = 'dataReads[1]', description = 'First Data Read',
    graphAttributes = {style = 'filled', fillcolor = 'green'}
}
memorys = {nn.Identity()()}
memorys[1]:annotate{
    name = 'memorys[1]', description = 'First Memory',
    graphAttributes = {style = 'filled', fillcolor = 'green'}
}
readWeightss = {nn.Identity()()}
readWeightss[1]:annotate{
    name = 'readWeightss[1]', description = 'First Read Weights',
    graphAttributes = {style = 'filled', fillcolor = 'green'}
}
writeWeightss = {nn.Identity()()}
writeWeightss[1]:annotate{
    name = 'writeWeightss[1]', description = 'First Write Weights',
    graphAttributes = {style = 'filled', fillcolor = 'green'}
}

for t = 1, TEMPORAL_HORIZON do
    inputs[t] = nn.Identity()()
    inputs[t]:annotate{
        name = 'inputs[' .. t .. ']', description = 'Input ' .. t,
        graphAttributes = {style = 'filled', fillcolor = 'green'}
    }

    out = ntms[t]{inputs[t], dataReads[t], memorys[t], readWeightss[t], writeWeightss[t]}
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

modelInputs = concatenateTables({inputs, {dataReads[1], memorys[1], readWeightss[1], writeWeightss[1]}})
modelOutputs = concatenateTables({outputs, {dataReads[TEMPORAL_HORIZON+1], memorys[TEMPORAL_HORIZON+1], readWeightss[TEMPORAL_HORIZON+1], writeWeightss[TEMPORAL_HORIZON+1]}})
model = nn.gModule(modelInputs, modelOutputs)
graph.dot(model.fg, 'Model', 'model')
