require 'nn'
require 'nngraph'
require 'modules/ScalarMulTable'
require 'modules/SmoothCosineSimilarity'
require 'modules/CircularConvolution'
require 'modules/PowTable'
require 'modules/OuterProd'

local function NTM(input_size, output_size, memory_slots, memory_size, controller_size, shift_size)
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
    local output = nn.Tanh()(nn.Linear(controller_size, output_size)(controllerOutput))
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

-- Example with graph
local ntm = NTM(4, 2, 10, 3, 5, 1)
graph.dot(ntm.fg, 'NTM', 'NTM')

return {NTM = NTM}
