require 'torch'
require 'nn'

require 'cutorch'
require 'cunn'

-- cutorch.setDevice(1)
dtype = 'torch.CudaTensor'


local inputSize = 5;
local dimensions = 2;
local clusters = 4;
local hiddenSize = 12;
local depth = 2;

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
   s:add(nn.MulConstant(.01))
   local r = nn.CMul(clusters)
   r.weight:fill(1) -- TODO
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
local function createSample()
   local numPerCluster = 5;
   
   -- Start from points in the low-dimensional space
   local points = torch.Tensor(numPerCluster * clusters, dimensions)
   local labels = torch.Tensor(numPerCluster * clusters, clusters):zero()
   local features = torch.Tensor(numPerCluster * clusters, inputSize)
   
   --Points at integer coordinates and some random points nearby
   for j = 1, clusters do
      local from = (j-1)*numPerCluster + 1
      points[from]:random(clusters * 2)
      labels[from][j] = 1
      
      for i = 2, numPerCluster do
	 local to = (j-1)*numPerCluster + i
	 -- print(from .. ":" .. to)
	 labels[to][j] = 1
	 for d = 1, dimensions do
	    points[to][d] = torch.normal(points[from][d], 0.1)
	 end
      end
   end
   
   -- Translate the points to features
   assert(inputSize == 5, "code supports inputSize = 5")
   for i = 1, numPerCluster * clusters do
      features[i][1] = points[i][1] + points[i][2]
      features[i][2] = points[i][1] - points[i][2]
      features[i][3] = 1 - points[i][1] + points[i][2] * 2
      features[i][4] = 2 - points[i][1] / 2 - points[i][2]
      features[i][5] = points[i][1] + torch.normal(0, 1)
   end
   
   return features:type(dtype), labels:type(dtype), points:type(dtype)
end

local function prettyprint(msg, tensor)
   print(msg)
   for i = 1, tensor:size(1) do
      for j = 1, tensor:size(2) do
	 io.write(string.format("% .2e", tensor[i][j]))
      end
      io.write("\n")
   end
end

local model = createModel()
local criterion = nn.AbsCriterion():type(dtype)
local x, y, points = createSample()

local function trainStep(x, y, learningRate)
   local pred = model:forward(x)
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   model:zeroGradParameters()
   model:backward(x, gradCriterion)
   model:updateParameters(learningRate)
   return err
end

local err
while true do
   for i = 1, 1000 do
      err = trainStep(x, y, 0.01)
   end
   print(err)
   prettyprint("sample embedded", points:t())
   prettyprint("modelled embedding", model:get(depth*2+1).output:t()) -- the last Linear layer outputting the embedded points
   prettyprint("distances", model:get(depth*2+2).output:t()) -- the Euclidean
   prettyprint("centres", model:get(depth*2+2).weight) -- the Euclidean
   prettyprint("mult by radii", model:get(depth*2+3).output:t()) -- the Euclidean
   print("radii")
   print(model:get(depth*2+4).weight) -- CMul
   prettyprint("square", model:get(depth*2+5).output:t())
   prettyprint("exp", model.output:t()) -- the pairwise probabilities of belonging to the clusters
   
   prettyprint("y", y:t())
   
end

-- print(mlp:get(1).weight)


