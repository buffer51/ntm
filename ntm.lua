require 'nn'
require 'nngraph'
require 'modules/ScalarMulTable'
require 'modules/SmoothCosineSimilarity'
require 'modules/CircularConvolution'
require 'modules/PowTable'

-- Parameters
local INPUT_SIZE = 4
local OUTPUT_SIZE = 2
local MEMORY_SLOTS = 10 -- Number of addressable slots in memory
local MEMORY_SIZE = 3 -- Size of each memory slot
local CONTROLLER_SIZE = 5
local SHIFT_SIZE = 1

input = nn.Identity()()
input:annotate{
    name = 'input', description = 'Input',
    graphAttributes = {style = 'filled', fillcolor = 'green'}
}
dataReadP = nn.Identity()()
dataReadP:annotate{
    name = 'dataReadP', description = 'Previous Data Read',
    graphAttributes = {style = 'filled', fillcolor = 'green'}
}
memoryP = nn.Identity()()
memoryP:annotate{
    name = 'memoryP', description = 'Previous Memory',
    graphAttributes = {style = 'filled', fillcolor = 'green'}
}
readWeightsP = nn.Identity()()
readWeightsP:annotate{
    name = 'readWeightsP', description = 'Previous Read Weights',
    graphAttributes = {style = 'filled', fillcolor = 'green'}
}

controllerOutput = nn.Tanh()(nn.CAddTable(){
    nn.Linear(INPUT_SIZE, CONTROLLER_SIZE)(input),
    nn.Linear(MEMORY_SIZE, CONTROLLER_SIZE)(dataReadP)
})
controllerOutput:annotate{
    name = 'controllerOutput', description = 'Controller Output',
    graphAttributes = {style = 'filled', fillcolor = 'orange'}
}

output = nn.Tanh()(nn.Linear(CONTROLLER_SIZE, OUTPUT_SIZE)(controllerOutput))
output:annotate{
    name = 'output', description = 'Output',
    graphAttributes = {style = 'filled', fillcolor = 'red'}
}


function newWeights(memoryP, controllerOutput, weightsP)
    local k = nn.Tanh()(nn.Linear(CONTROLLER_SIZE, MEMORY_SIZE)(controllerOutput))
    local beta = nn.SoftPlus()(nn.Linear(CONTROLLER_SIZE, 1)(controllerOutput))
    local contentWeights = nn.SoftMax()(nn.ScalarMulTable(){nn.SmoothCosineSimilarity(){memoryP, k}, beta})

    local g = nn.Sigmoid()(nn.Linear(CONTROLLER_SIZE, 1)(controllerOutput))
    local locationWeights = nn.CAddTable(){
        nn.ScalarMulTable(){contentWeights, g},
        nn.ScalarMulTable(){weightsP, nn.AddConstant(1)(nn.MulConstant(-1)(g))}
    }

    local s = nn.SoftMax()(nn.Linear(CONTROLLER_SIZE, 2 * SHIFT_SIZE + 1)(controllerOutput))
    local shiftedWeights = nn.CircularConvolution(){locationWeights, s}

    local gamma = nn.AddConstant(1)(nn.SoftPlus()(nn.Linear(CONTROLLER_SIZE, 1)(controllerOutput)))
    local sharpenedWeights = nn.Normalize(1)(nn.PowTable(){shiftedWeights, gamma})

    return sharpenedWeights
end

readWeights = newWeights(memoryP, controllerOutput, readWeightsP)
readWeights:annotate{
    name = 'readWeights', description = 'Read Weights',
    graphAttributes = {style = 'filled', fillcolor = 'orange'}
}

module = nn.gModule({input, dataReadP, memoryP, readWeightsP}, {output, readWeights})
graph.dot(module.fg, 'NTM', 'NTM')
