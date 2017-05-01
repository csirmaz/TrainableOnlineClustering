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
local OptimState = {learningRate = 0.01}

local function prettyprint(msg, tensor)
   print(msg)
   for i = 1, tensor:size(1) do
      for j = 1, tensor:size(2) do
         io.write(string.format("% .3e", tensor[i][j]))
      end
      io.write("\n")
   end
end

local function createTestModel()
   local s = nn.Sequential()
   s:add(nn.Linear(inputSize, hiddenSize))
   for i = 1, depth do
      s:add(nn.ReLU())
      if i < depth then
         s:add(nn.Linear(hiddenSize, hiddenSize)) -- layer 3 ...
      end
   end
   s:add(nn.ReLU())
   s:add(nn.Linear(hiddenSize, clusters))
   s:add(nn.SoftMax())
   return s:type(dtype)
end

local function createModel()
   
   local s = nn.Sequential()
   
   -- The embedding network
   -- ---------------------

   -- inputSize == ?
   -- outputSize == dimensions
   
   if depth == 0 then
      -- http://www.epcsirmaz.com/torch/torch_nn-simple_layers-linear.html
      s:add(nn.Linear(inputSize, dimensions)) -- layer 1
   else
   
      s:add(nn.Linear(inputSize, hiddenSize)) -- layer 1
      
      for i = 1, depth do
         -- http://www.epcsirmaz.com/torch/torch_nn-transfer_function_layers-relu.html
         s:add(nn.ReLU()) -- layer 2 ... depth*2
         if i < depth then
            s:add(nn.Linear(hiddenSize, hiddenSize)) -- layer 3 ...
         end
      end   
      s:add(nn.Linear(hiddenSize, dimensions)) -- layer depth*2+1
   end
   
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
   
   -- in: (batchSize x clusters) out: (batchSize x clusters)
   -- parameters: clusters scalars
   -- we learn just a single factor per cluster, which is 1/radius
   -- http://www.epcsirmaz.com/torch/torch_nn-simple_layers-mul.html
   
   -- s:add(nn.MulConstant(.01))

   local r = nn.CMul(clusters)
   -- r.weight:fill(1) -- TODO
   s:add(r)
   
   -- in: (batchSize x clusters) out: (batchSize x clusters)
   -- parameters: none
   -- http://www.epcsirmaz.com/torch/torch_nn-simple_layers-square.html
   s:add(nn.Square())

   -- in: (batchSize x clusters) out: (batchSize x clusters)
   -- parameters: none
   -- http://www.epcsirmaz.com/torch/torch_nn-transfer_function_layers-mulconstant.html
   s:add(nn.MulConstant(-1.0))
   
   -- in: (batchSize x clusters) out: (batchSize x clusters)
   -- parameters: none
   -- http://www.epcsirmaz.com/torch/torch_nn-simple_layers-exp.html
   s:add(nn.Exp())
   
   return s:type(dtype)
end

-- Create some sample data to test the network
-- Returns:
--   x - input to the model
--   y -
--   targetlabels
local function createSample()
   
   local data = require '../LearnPixels/output/LearnPixels.lua'
   local datasize = #data
   -- datasize = 15 -- TODO

   local x = torch.Tensor(datasize, inputSize)
   -- local y = torch.Tensor(datasize) -- target is class index
   local y = torch.Tensor(datasize, clusters):zero()
   local targetlabels = torch.Tensor(datasize, 1)

   for i = 1, datasize do
      local d = data[i]
      
      for j, pixel in ipairs(d.img) do
         x[i][j] = pixel
      end
      
      -- d.classnum = (i > clusters/2 and clusters or i)-1 -- TODO
      y[i][d.classnum+1] = 1
      -- y[i] = d.classnum+1 -- target is class index
      targetlabels[i][1] = d.classnum+1
   end
   
   print("x"); print(x)
   print("y"); print(y)
   return x:type(dtype), y:type(dtype), targetlabels:long()
end


local model = createTestModel()
local criterion = nn.MSECriterion():type(dtype) -- nn.AbsCriterion():type(dtype)
local x, y, targetlabels = createSample()
local rep = 0

local params, gradParams = model:getParameters()

function debug(err)
   rep = rep + 1
   if rep < 10000 then return end
   rep = 0
   
   local output = model.output
   local _, predictions = output:float():sort(2, true) -- class indices ordered by probability
   local correct = predictions:eq(
      targetlabels:expandAs(output)
   )
   local correctratio = correct:narrow(2, 1, 1):sum() / output:size(1)

   print(err, correctratio, OptimState.learningRate)
   -- prettyprint("sample embedded", points:t())
   -- prettyprint("modelled embedding", model:get(depth*2+1).output:t()) -- the last Linear layer outputting the embedded points
   -- prettyprint("distances", model:get(depth*2+2).output:t()) -- the Euclidean
   -- prettyprint("centres", model:get(depth*2+2).weight) -- the Euclidean
   
   -- prettyprint("mult by radii", model:get(depth*2+4).output:t())
   -- print("radii")
   -- print(model:get(depth*2+4).weight) -- CMul
   
   -- prettyprint("res", model.output)
   -- prettyprint("exp", model.output:t()) -- the pairwise probabilities of belonging to the clusters
   
   -- prettyprint("y", y:t())
   
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


