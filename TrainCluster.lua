require 'torch'
local nn = require 'nn'

local inputSize = 5;
local dimensions = 2;
local clusters = 4;

local function createModel()

   local hiddenSize = 12;
   local depth = 2;
   
   local s = nn.Sequential()
   
   -- The embedding network
   -- ---------------------

   -- inputSize == ?
   -- outputSize == dimensions
   
   -- http://www.epcsirmaz.com/torch/torch_nn-simple_layers-linear.html
   s:add(nn.Linear(inputSize, hiddenSize))
   
   -- http://www.epcsirmaz.com/torch/torch_nn-transfer_function_layers-relu.html
   s:add(nn.ReLU())
   for i = 1, depth-1 do
      s:add(nn.Linear(hiddenSize, hiddenSize))
      s:add(nn.ReLU())
   end   
   s:add(nn.Linear(hiddenSize, dimensions))
      
   
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
   s:add(nn.Euclidean(dimensions, clusters))
   
   -- in: (batchSize x clusters) out: (batchSize x clusters)
   -- parameters: clusters scalars
   -- we learn just a single factor per cluster, which is 1/radius
   -- http://www.epcsirmaz.com/torch/torch_nn-simple_layers-mul.html
   s:add(nn.CMul(clusters))
   
   
   -- http://www.epcsirmaz.com/torch/torch_nn-transfer_function_layers-tanh.html
   -- TODO alternatively, use nn.Tanh
   

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
   
   return s
end

-- Create some sample data to test the network
local function createSample()
   local numPerCluster = 10;
   
   -- Start from points in the low-dimensional space
   local points = torch.Tensor(numPerCluster * clusters, dimensions)
   local labels = torch.Tensor(numPerCluster * clusters, clusters):zero()
   local features = torch.Tensor(numPerCluster * clusters, inputSize)
   
   --Points at integer coordinates and some random points nearby
   for j = 1, clusters do
      local from = (j-1)*numPerCluster + 1
      points[from]:random(clusters)
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
   
   return features, labels
end

local model = createModel()
local criterion = nn.AbsCriterion()
local x, y = createSample()

local function trainStep(x, y, learningRate)
   local pred = model:forward(x)
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   model:zeroGradParameters()
   model:backward(x, gradCriterion)
   model:updateParameters(learningRate)
   return err
end

trainStep(
   torch.Tensor(inputSize),
   torch.Tensor(clusters),
   0.01
)

for i = 1, 100000 do
   local err = trainStep(x, y, 0.01)
   if i % 1000 == 0 then
      print(err)
   end
end

-- print(mlp:get(1).weight)


