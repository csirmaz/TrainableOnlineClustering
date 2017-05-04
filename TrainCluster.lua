require 'torch'
require 'nn'
require 'optim'
require 'gnuplot'

require 'cutorch'
require 'cunn'

-- cutorch.setDevice(1)
dtype = 'torch.CudaTensor'

local InputSize = 25; -- size of input
local Dimensions = 3; -- number of dimensions of the embedding space
local Clusters = false; -- number of clusters / classes
local HiddenSize = 32; -- size of hidden layers
local Depth = 4; -- number of hidden layers
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

   -- InputSize == ?
   -- outputSize == Dimensions
   
   if Depth == 0 then
      -- http://www.epcsirmaz.com/torch/torch_nn-simple_layers-linear.html
      s:add(nn.Linear(InputSize, Dimensions))
   else
   
      s:add(nn.Linear(InputSize, HiddenSize))
      
      for i = 1, Depth do
         -- http://www.epcsirmaz.com/torch/torch_nn-transfer_function_layers-relu.html
         s:add(nn.ReLU())
         if i < Depth then
            s:add(nn.Linear(HiddenSize, HiddenSize))
         end
      end   
      s:add(nn.Linear(HiddenSize, Dimensions))
   end
   
   layerIndex["embedout"] = s:size()
   
   -- The clustering head
   -- -------------------
   
   -- InputSize == Dimensions
   -- outputSize == Clusters

   -- Here we implement clustering based on the following distance function:
   -- e^(-distance^2/radius^2)
   
   -- calculate distance to cluster centers
   -- in: (batchSize x Dimensions) out: (batchSize x Clusters)
   -- parameters: Dimensions x Clusters coordinates
   -- http://www.epcsirmaz.com/torch/torch_nn-simple_layers-euclidean.html -- https://github.com/torch/nn/blob/master/doc/simple.md#euclidean
   s:add(nn.Euclidean(Dimensions, Clusters))
   layerIndex["euclid"] = s:size()
   
   return s:type(dtype)
end

-- Create some sample data to test the network
-- Returns:
--   number of clusters
--   x - input to the model
--   y - -1/1 encoding
--   targetlabels
local function createSample(filename, create_y)
   
   local data, numclusters = require(filename)
   local datasize = #data

   local x = torch.Tensor(datasize, InputSize)
   local y = torch.Tensor(datasize, numclusters):fill(-1)
   local targetlabels = torch.Tensor(datasize, 1)

   for i = 1, datasize do
      local d = data[i]
      
      for j, pixel in ipairs(d.img) do
         x[i][j] = pixel
      end
      
      if create_y then
        y[i][d.classnum+1] = 1
      end
      targetlabels[i][1] = d.classnum+1
   end
   
   return numclusters, x:type(dtype), y:type(dtype), targetlabels:long()
end

-- --------------------
-- TRAIN
-- --------------------

local numclusters, x, y, targetlabels = createSample('../LearnPixels/output/train.lua', true)
Clusters = numclusters

local model = createModel()

-- https://github.com/torch/nn/blob/master/doc/criterion.md#hingeembeddingcriterion
local criterion = nn.HingeEmbeddingCriterion(1.0):type(dtype)

local rep = 0

local keep_training = true

local params, gradParams = model:getParameters()

function evaluate(err, prediction)
   rep = rep + 1
   if rep < 1000 then return end
   rep = 0
   
   -- prediction == model.output
   local _, predictions = prediction:float():sort(2) -- class indices ordered by distance
   local correct = predictions:eq(
      targetlabels:expandAs(prediction)
   )
   local correctratio = correct:narrow(2, 1, 1):sum() / prediction:size(1)

   -- prettyprint("modelled embedding", model:get(layerIndex["embedout"]).output)

   -- prettyprint("distances", model:get(layerIndex["euclid"]).output)
   -- prettyprint("centres", model:get(layerIndex["euclid"]).weight)

   -- prettyprint("out", output)
   -- prettyprint("pred", predictions)
   
   print("Error", err, "Correct%", correctratio*100)
   
   if correctratio > 0.99 then keep_training = false end
end

function feval(params)
   gradParams:zero()
   local pred = model:forward(x)
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   model:backward(x, gradCriterion)
   
   evaluate(err, pred)

   return err, gradParams
end

while keep_training do
   optim.adam(feval, params, OptimState)
end

local ClusterCenters = model:get(layerIndex["euclid"]).weight:float()

-- --------------------
-- TEST
-- --------------------

-- Run the model on the test set and see how it embeds the images
-- in the low-dimensional space
function test()
   local _, input, _, testlabels = createSample('../LearnPixels/output/test.lua', false)
   model:forward(input)
   local embedding = model:get(layerIndex["embedout"]).output:float()
   
   local plotdata = {}

   for i = 1, input:size(1) do
      local label = testlabels[i][1]
      local point = embedding[i]
      
      if not plotdata[label] then
         plotdata[label] = {("Class %d"):format(label), {}, {}, {}}
      end
      
      table.insert(plotdata[label][2], point[1]) -- x
      table.insert(plotdata[label][3], point[2]) -- y
      table.insert(plotdata[label][4], point[3]) -- z
      
   end
   
   local plotargs = {}
   for _, v in pairs(plotdata) do
      table.insert(plotargs, {
         v[1],
         torch.Tensor(v[2]),
         torch.Tensor(v[3]),
         torch.Tensor(v[4])
      })
   end
   
   gnuplot.scatter3(unpack(plotargs))

end

test()

