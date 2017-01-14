require 'nn'
local ntm = require 'ntm'
local model_utils = require 'model_utils'

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
memorys = {nn.Identity()()}
readWeightss = {nn.Identity()()}
writeWeightss = {nn.Identity()()}

for t = 1, TEMPORAL_HORIZON do
    inputs[t] = nn.Identity()()

    out = ntms[t]{inputs[t], dataReads[t], memorys[t], readWeightss[t], writeWeightss[t]}
    outputs[t] = out[1]
    dataReads[t+1] = out[2]
    memorys[t+1] = out[3]
    readWeightss[t+1] = out[4]
    writeWeightss[t+1] = out[5]
end

-- TODO: Create the model
-- model = nn.gModule({inputs, dataReads, memorys, readWeightss, writeWeightss}, outputs)
-- graph.dot(model.fg, 'Model', 'model')
