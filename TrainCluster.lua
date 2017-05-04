require 'torch'
require 'nn'
require 'optim'
require 'gnuplot'

require 'cutorch'
require 'cunn'

-- cutorch.setDevice(1)
dtype = 'torch.CudaTensor'

local InputSize = 784; -- size of input
local Dimensions = 3; -- number of dimensions of the embedding space
local Clusters = false; -- number of clusters / classes
local HiddenSize = 128; -- size of hidden layers
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
local function createSample(filename, create_y)
   print("Loading "..filename)
   local d = require(filename)
   return d.numclusters, d.x:type(dtype), d.y:type(dtype), d.targetlabels:long()
end

-- --------------------
-- TRAIN
-- --------------------

local numclusters, x, y, targetlabels = createSample('data/train.lua')
local _, test_x, test_y, test_targetlabels = createSample('data/test.lua')

Clusters = numclusters

local model = createModel()
model:training()

-- https://github.com/torch/nn/blob/master/doc/criterion.md#hingeembeddingcriterion
local criterion = nn.HingeEmbeddingCriterion(1.0):type(dtype)

local rep = 0

local keep_training = true

local params, gradParams = model:getParameters()

-- Called by the training function
function evaluate(train_err, train_prediction)
   rep = rep + 1
   if rep < 500 then return end
   rep = 0
   
   model:evaluate()
   
   local test_pred = model:forward(test_x)
   model:zeroGradParameters()
   local test_err = criterion:forward(test_pred, test_y)
   -- TODO zero this too?
   
   local _, predictions = test_pred:float():sort(2) -- class indices ordered by distance
   local correct = predictions:eq(
      test_targetlabels:expandAs(test_pred)
   )
   local correctratio = correct:narrow(2, 1, 1):sum() / test_pred:size(1)

   print("Train error", train_err, "Test error", test_err, "Test Correct%", correctratio*100)
   
   if correctratio > 0.99 then keep_training = false end
   model:training()
end

-- The training function
function feval(params)
   gradParams:zero()
   local pred = model:forward(x)
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   model:backward(x, gradCriterion)
   
   evaluate(err, pred)

   return err, gradParams
end

-- The training loop
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
   local _, input, _, testlabels = createSample('data/test.lua')
   model:evaluate()
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

