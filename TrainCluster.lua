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

-- Create a simple ReLU network for testing.
-- Use MSECriterion with this network.
local function createTestModel()
   local s = nn.Sequential()
   s:add(nn.Linear(inputSize, hiddenSize))
   for i = 1, depth do
      s:add(nn.ReLU())
      if i < depth then
         s:add(nn.Linear(hiddenSize, hiddenSize))
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
   
   -- in: (batchSize x clusters) out: (batchSize x clusters)
   -- parameters: clusters scalars
   -- we learn just a single factor per cluster, which is 1/radius
   -- http://www.epcsirmaz.com/torch/torch_nn-simple_layers-mul.html
   
   -- -- s:add(nn.MulConstant(10))

   -- local r = nn.CMul(clusters)
   -- r.weight:fill(0.01) -- TODO
   -- s:add(r)
   
   -- in: (batchSize x clusters) out: (batchSize x clusters)
   -- parameters: none
   -- http://www.epcsirmaz.com/torch/torch_nn-simple_layers-square.html
   -- -- s:add(nn.Square())

   -- in: (batchSize x clusters) out: (batchSize x clusters)
   -- parameters: none
   -- http://www.epcsirmaz.com/torch/torch_nn-transfer_function_layers-mulconstant.html
   -- -- s:add(nn.MulConstant(-1.0))
   
   -- in: (batchSize x clusters) out: (batchSize x clusters)
   -- parameters: none
   -- http://www.epcsirmaz.com/torch/torch_nn-simple_layers-exp.html
   -- -- s:add(nn.Exp())

   -- s:add(nn.SoftMax())
   
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
   local y = torch.Tensor(datasize, clusters):fill(-1)
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


-- ===========================================================================================

local HingeEmbeddingCriterion, parent = torch.class('nn.MyHingeEmbeddingCriterion', 'nn.Criterion')

function HingeEmbeddingCriterion:__init(margin)
   parent.__init(self)
   self.margin = margin or 1
   self.sizeAverage = true
end 
 
function HingeEmbeddingCriterion:updateOutput(input,y)
   self.buffer = self.buffer or input.new()

   self.buffer:resizeAs(input):copy(input)
   self.buffer[torch.eq(y, -1)] = 0
   self.output = self.buffer:sum()
   
   self.buffer:fill(self.margin):add(-1, input)
   self.buffer:cmax(0)
   self.buffer[torch.eq(y, 1)] = 0
   self.output = self.output + self.buffer:sum()
   
   if (self.sizeAverage == nil or self.sizeAverage == true) then 
      self.output = self.output / input:nElement()
   end

   return self.output -- error
end

function HingeEmbeddingCriterion:updateGradInput(input, y)
   self.gradInput:resizeAs(input):copy(y)
   self.gradInput[torch.cmul(torch.eq(y, -1), torch.gt(input, self.margin))] = 0
   
   if (self.sizeAverage == nil or self.sizeAverage == true) then
      self.gradInput:mul(1 / input:nElement())
   end
      
   return self.gradInput 
end

-- ===========================================================================================

local model = createModel()
-- local criterion = nn.MSECriterion():type(dtype)
-- local criterion = nn.AbsCriterion():type(dtype)

-- https://github.com/torch/nn/blob/master/doc/criterion.md#hingeembeddingcriterion
local criterion = nn.MyHingeEmbeddingCriterion(1.0):type(dtype)

local x, y, targetlabels = createSample()
local rep = 0

local params, gradParams = model:getParameters()

function debug(err)
   rep = rep + 1
   if rep < 1000 then return end
   rep = 0
   
   local output = model.output
   local _, predictions = output:float():sort(2) -- class indices ordered by probability
   local correct = predictions:eq(
      targetlabels:expandAs(output)
   )
   local correctratio = correct:narrow(2, 1, 1):sum() / output:size(1)

   
   -- prettyprint("modelled embedding", model:get(layerIndex["embedout"]).output:t()) -- the last Linear layer outputting the embedded points

   -- prettyprint("distances", model:get(layerIndex["euclid"]).output)
   -- prettyprint("centres", model:get(layerIndex["euclid"]).weight)

   -- prettyprint("out", output)
   -- prettyprint("pred", predictions)
   
   print(err, correctratio, OptimState.learningRate)
   
   -- os.execute("sleep 1")
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


