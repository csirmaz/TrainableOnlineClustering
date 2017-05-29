-- For the parent class, see https://raw.githubusercontent.com/torch/nn/master/Euclidean.lua

local Euclidean, parent = torch.class('nn.LazyEuclidean', 'nn.Euclidean')

function Euclidean:__init(inputSize, outputSize, learningRateFactor)
   parent.__init(self, inputSize, outputSize)
   self.learningRateFactor = learningRateFactor
end

-- Apply a factor to the learning rate to make the layer less responsive
-- +----+
-- |NOTE| optim functions do not call updateParameters, so this override has no effect.
-- +----+
function Euclidean:updateParameters(learningRate)
   parent.updateParameters(self, learningRate * self.learningRateFactor)
end

-- Distribute the points better initially
function Euclidean:reset()
   local max = self.weight:size(2)
   self.weight:random(max):csub(max/2)
   while self:MinCenterDistance() < self.weight:size(1) do
      self.weight:random(max):csub(max/2)
   end
end

-- Return the minimum square distance between the centers
function Euclidean:MinCenterDistance()
   local min
   local wt = self.weight:t()
   local wts = wt:size(1)
   for i = 1, wts-1 do
      for j = i+1, wts do
         local d = torch.add(wt[i], -1, wt[j]):pow(2):sum()
         if (not min) or (min > d) then
            min = d
         end
      end
   end
   return min
end

