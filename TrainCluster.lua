require 'torch'
require 'nn'
require 'optim'

require 'cutorch'
require 'cunn'

-- cutorch.setDevice(1)
dtype = 'torch.CudaTensor'

local inputSize = 25; -- size of input
local dimensions = 4; -- number of dimensions of the embedding space
local clusters = 14; -- number of clusters / classes
local hiddenSize = 32; -- size of hidden layers
local depth = 3; -- number of hidden layers
local OptimState = {learningRate = 0.01} -- 0.01
local layerIndex = {}

local function prettyprint(msg, tensor)
   print(msg)
   for i = 1, tensor:size(1) do
      for j = 1, tensor:size(2) do
         io.write(string.format("% .3e", tensor[i][j]))
      end
      io.write("\n")
   end
end

local function createModel()
   
   local s = nn.Sequential()
   
   -- The embedding network
   -- ---------------------

   -- inputSize == ?
   -- outputSize == dimensions
   
   if depth == 0 then
      -- http://www.epcsirmaz.com/torch/torch_nn-simple_layers-linear.html
      s:add(nn.Linear(inputSize, dimensions))
   else
   
      s:add(nn.Linear(inputSize, hiddenSize))
      
      for i = 1, depth do
         -- http://www.epcsirmaz.com/torch/torch_nn-transfer_function_layers-relu.html
         s:add(nn.ReLU()) -- layer 2 ... depth*2
         if i < depth then
            s:add(nn.Linear(hiddenSize, hiddenSize))
         end
      end   
      s:add(nn.Linear(hiddenSize, dimensions))
   end
   
   layerIndex["embedout"] = s:size()
   
   -- The clustering head
   -- -------------------
   
   -- inputSize == dimensions
   -- outputSize == clusters

   -- Here we implement clustering based on the following distance function:
   -- e^(-distance^2/radius^2)
   
   -- calculate distance to cluster centers
   -- in: (batchSize x dimensions) out: (batchSize x clusters)
   -- parameters: dimensions x clusters coordinates
   -- http://www.epcsirmaz.com/torch/torch_nn-simple_layers-euclidean.html -- https://github.com/torch/nn/blob/master/doc/simple.md#euclidean
   s:add(nn.Euclidean(dimensions, clusters)) -- layer depth*2+2
   layerIndex["euclid"] = s:size()
   
   return s:type(dtype)
end

-- Create some sample data to test the network
-- Returns:
--   x - input to the model
--   y - -1/1 encoding
--   targetlabels
local function createSample()
   
   local data = require '../LearnPixels/output/LearnPixels.lua'
   local datasize = #data

   local x = torch.Tensor(datasize, inputSize)
   local y = torch.Tensor(datasize, clusters):fill(-1)
   local targetlabels = torch.Tensor(datasize, 1)

   for i = 1, datasize do
      local d = data[i]
      
      for j, pixel in ipairs(d.img) do
         x[i][j] = pixel
      end
      
      y[i][d.classnum+1] = 1
      targetlabels[i][1] = d.classnum+1
   end
   
   return x:type(dtype), y:type(dtype), targetlabels:long()
end


local model = createModel()

-- https://github.com/torch/nn/blob/master/doc/criterion.md#hingeembeddingcriterion
local criterion = nn.HingeEmbeddingCriterion(1.0):type(dtype)

local x, y, targetlabels = createSample()

local rep = 0

local params, gradParams = model:getParameters()

function debug(err)
   rep = rep + 1
   if rep < 1000 then return end
   rep = 0
   
   local output = model.output
   local _, predictions = output:float():sort(2) -- class indices ordered by distance
   local correct = predictions:eq(
      targetlabels:expandAs(output)
   )
   local correctratio = correct:narrow(2, 1, 1):sum() / output:size(1)

   -- prettyprint("modelled embedding", model:get(layerIndex["embedout"]).output:t())

   -- prettyprint("distances", model:get(layerIndex["euclid"]).output)
   -- prettyprint("centres", model:get(layerIndex["euclid"]).weight)

   -- prettyprint("out", output)
   -- prettyprint("pred", predictions)
   
   print("Error", err, "Correct%", correctratio*100)
end

function feval(params)
   gradParams:zero()
   local pred = model:forward(x)
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   model:backward(x, gradCriterion)
   
   debug(err)

   return err, gradParams
end

while true do
   optim.adam(feval, params, OptimState)
end


