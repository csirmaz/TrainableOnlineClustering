require 'torch'
local nn = require 'nn'


local function createModel()

   local inputSize = 5;
   local hiddenSize = 12;
   local depth = 4;
   local dimensions = 2;
   local clusters = 4;
   
   local s = nn.Sequential()
   
   -- The embedding network
   -- ---------------------

   -- inputSize == ?
   -- outputSize == dimensions
   
   -- http://www.epcsirmaz.com/torch/torch_nn-simple_layers-linear.html
   s:add(nn.Linear(inputSize, hiddenSize));
   
   -- http://www.epcsirmaz.com/torch/torch_nn-transfer_function_layers-relu.html
   s:add(nn.ReLU)
   for i = 1, depth-1 do
      s:add(nn.Linear(hiddenSize, hiddenSize));
      s:add(nn.ReLU)
   end   
   s:add(nn.Linear(hiddenSize, dimensions));
   
   
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
   s:add(nn.MulConstant(-1.0)
   
   -- in: (batchSize x clusters) out: (batchSize x clusters)
   -- parameters: none
   -- http://www.epcsirmaz.com/torch/torch_nn-simple_layers-exp.html
   s:add(nn.Exp())
   
   return s;
end
