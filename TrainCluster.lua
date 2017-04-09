require 'torch'
require 'nn'


local layer, parent = torch.class('nn.TrainCluster', 'nn.Module')

-- see https://github.com/torch/nn/blob/master/Module.lua
-- https://github.com/torch/nn/blob/master/doc/module.md#nn.Module

-- Based on https://github.com/torch/nn/blob/master/Euclidean.lua


-- inputSize == dimensions
-- outputSize == clusters
function layer:__init(dimensions, clusters)
   parent.__init(self)

   self.dimensions = dimensions;
   self.clusters = clusters;
   
   -- The midpoints of the clusters
   self.cpoints = torch.Tensor(dimensions, clusters)
   self.cpointsGrad = torch.Tensor(dimensions, clusters):zero()
   
   -- The radii of the clusters
   self.radii = torch.Tensor(clusters)
   self.radiiGrad = torch.Tensor(clusters):zero()
   
   self:_reset()
end

function layer:parameters()
   return {self.cpoints, self.radii}, {self.cpointsGrad, self.radiiGrad}
end

function layer:_reset()
   self.cpoints:normal(0, 10)
   self.radii:normal(0, 10)
end

-- forward
-- input: batchsize x dimensions
-- output: batchsize x clusters
function layer:updateOutput(input)
   
   

   if input:dim() == 1 then -- no batch, just a single point
   
      -- self._input becomes a view of input with an extra dimension (dimensions x 1)
      -- http://www.epcsirmaz.com/torch/torch-tensor-manipulating_the_tensor_view-view.html -- https://github.com/torch/torch7/blob/master/doc/tensor.md#result-viewresult-tensor-sizes
      view(self._input, input, self.dimensions, 1)
      
      -- create a new view where the 1-long dimension is repeated (dimensions x clusters)
      -- http://www.epcsirmaz.com/torch/torch-tensor-expanding_replicating_squeezing_tensors-expandas.html -- https://github.com/torch/torch7/blob/master/doc/tensor.md#result-expandasresult-tensor
      -- http://www.epcsirmaz.com/torch/torch-tensor-expanding_replicating_squeezing_tensors-expand.html -- https://github.com/torch/torch7/blob/master/doc/tensor.md#result-expandresult-sizes
      self._expand:expandAs(self._input, self.cpoints)
      
      -- copy the data
      -- http://www.epcsirmaz.com/torch/torch-tensor-copying_and_initializing-copy.html
      self._repeat:resizeAs(self._expand):copy(self._expand)
      
      -- subtract the midpoints of the clusters (dimensions x clusters)
      -- http://www.epcsirmaz.com/torch/torch-math_functions-basic_operations-add.html
      self._repeat:add(-1, self.cpoints)

      -- 2-norm (Euclidean length) of _repeat over dimension 1 (dimensions) -> (1 x clusters)
      -- http://www.epcsirmaz.com/torch/torch-math_functions-matrix_wide_operations-norm.html
      self.output:norm(self._repeat, 2, 1)
      
      -- resize to (clusters)
      self.output:resize(self.clusters)
      
      -- TODO Apply Gaussian
      
   elseif input:dim() == 2 then -- there is a batch, input::(batchSize x dimensions)

      local batchSize = input:size(1)
      
      -- self._input becomes a view of input with an extra dimension (batchSize x dimensions x 1)
      -- http://www.epcsirmaz.com/torch/torch-tensor-manipulating_the_tensor_view-view.html -- https://github.com/torch/torch7/blob/master/doc/tensor.md#result-viewresult-tensor-sizes
      view(self._input, input, batchSize, self.dimensions, 1)

      -- create a new view where the 1-long dimension is repeated (batchSize x dimensions x clusters)
      -- http://www.epcsirmaz.com/torch/torch-tensor-expanding_replicating_squeezing_tensors-expand.html -- https://github.com/torch/torch7/blob/master/doc/tensor.md#result-expandresult-sizes
      self._expand:expand(self._input, batchSize, self.dimensions, self.clusters)

      -- copy the data
      -- make the expanded tensor contiguous (requires lots of memory)
      self._repeat:resizeAs(self._expand):copy(self._expand)
      
      -- a view of cpoints (1 x dimensions x clusters)
      self._cpoints:view(self.cpoints, 1, self.dimensions, self.clusters)

      -- create a new view where the 1-long dimension is repeated (batchSize x dimensions x clusters)
      -- http://www.epcsirmaz.com/torch/torch-tensor-expanding_replicating_squeezing_tensors-expandas.html -- https://github.com/torch/torch7/blob/master/doc/tensor.md#result-expandasresult-tensor
      -- http://www.epcsirmaz.com/torch/torch-tensor-expanding_replicating_squeezing_tensors-expand.html -- https://github.com/torch/torch7/blob/master/doc/tensor.md#result-expandresult-sizes      
      self._cpoints_expand:expandAs(self._cpoints, self._repeat)
      
      if torch.type(input) == 'torch.CudaTensor' then
         -- requires lots of memory, but minimizes cudaMallocs and loops
	 -- copy the data (batchSize x dimensions x clusters)
         self._cpoints_repeat:resizeAs(self._cpoints_expand):copy(self._cpoints_expand)
	 
	 -- subtract the midpoints of the clusters (batchSize x dimensions x clusters)
	 -- http://www.epcsirmaz.com/torch/torch-math_functions-basic_operations-add.html
         self._repeat:add(-1, self._cpoints_repeat)
      else
	 -- subtract the midpoints of the clusters (batchSize x dimensions x clusters)
	 -- http://www.epcsirmaz.com/torch/torch-math_functions-basic_operations-add.html
         self._repeat:add(-1, self._cpoints_expand)
      end
      
      -- 2-norm (Euclidean length) of _repeat over dimension 2 (dimensions) -> (batchSize x 1 x clusters)
      -- http://www.epcsirmaz.com/torch/torch-math_functions-matrix_wide_operations-norm.html
      self.output:norm(self._repeat, 2, 2)

      -- resize to (batchSize x clusters)
      self.output:resize(batchSize, self.clusters)
      
      -- TODO Apply Gaussian
   else
      error"1D or 2D input expected"
   end
   
   return self.output
end

-- backward 1/2
-- Computing the gradient of the module with respect to its own input.
function layer:updateGradInput(input, gradOutput)
   return self.gradInput
end

-- backward 2/2
-- Computing the gradient of the module with respect to its own parameters.
-- The module is expected to accumulate the gradients with respect to the parameters in some variable.
-- scale is a scale factor that is multiplied with the gradParameters before being accumulated.
function layer:accGradParameters(input, gradOutput, scale)
end

