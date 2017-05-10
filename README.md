# TrainableOnlineClustering

Exploring recognizing unseen clusters with a neural network in an on-line
setting

## The problem

Given an initial, labeled batch of data, we aim to train a system
that is capable of classifying a stream of data and recognizing new classes
it has not yet seen.

## The idea

We aim to solve this problem by training an "embedding" neural net
to embed the input data points in a low dimensional space
(that is, do dimension reduction from
the high number of input features to a point in a low-dimensional Euclidean space)
in a way that points representing inputs belonging to the same class
are close to each other, and far from the points related to other classes.

We do this by adding a "clustering head" to the neural net, which
learns the centers of clusters related to each known class.
The embedding net and the clustering head are trained together on the labeled
data.

Then, when the system is used to classify the stream of data,
the already trained embedding network is used to embed each incoming data point in
the low-dimensional space, where similar inputs will hopefully be close to
each other, and dissimilar inputs be far from each other,
making it possible to use classic clustering techniques to identify
brand new clusters arising from a new class of inputs the system has not
previously seen (and better than by using a linear mapping e.g. PCA).

## The experiment

The system is implemented in Torch/nn,
using a modified
[`nn.Euclidean`](http://www.epcsirmaz.com/torch/torch_nn-simple_layers-euclidean.html)
as the clustering layer, and
[`nn.HingeEmbeddingCriterion`](http://www.epcsirmaz.com/torch/torch_nn-criterions-hingeembeddingcriterion.html)
as the criterion.
This combination implements the idea perfectly, as this criterion
will attract a point to its cluster center with a constant force regardless
of the distance, but will only repel a point from other cluster centers up
to a set distance.

I used part of the MNIST handwritten digit dataset for the experiment.
Each image in the dataset has 784 pixels.
The embedding network is a very simple fully connected feed-forward network
with 7 layers of ReLUs; 520 on each.
The output of the embedding network is 3 numbers, so each image is
represented by a point in 3D.

I needed to modify the `nn.Euclidean` layer to ensure that the initial
cluster centers are sufficiently far from each other, and to reduce its
learning rate relative to the rest of the network.
These were essential in getting the network able to learn the data.

I trained the whole network on 8 classes only, the ones for the digits
`0,1,2,4,5,6,8,9`. This dataset was split into a training dataset and
a testing dataset using the ratio 70/30.

Once the network correctly predicted at least 93% of the test data,
the embedding network was used to map some (10%) of the images for digits
`3` and `7` into the 3D space, along with a sample (1%) of images from the
other 8 classes. Please see the result below.

![Embedding results](https://raw.github.com/wiki/csirmaz/TrainableOnlineClustering/embedding.png)

It looks quite promising that the images for `3` (purple triangle) are
largely separated from those for `7` (orange diamond). Unfortunately,
they are not localized in a spherical cluster, but spread out touching
other clusters.
Training on more clusters may help by crowding the 3D space, as well
as training the network on inputs that do not need to be collected into a
cluster, but moved away from existing clusters.
While this experiment uses images, the same clustering idea could be used
with other types of data, including text, with a LSTM/GRU embedding network.

## Running

- Download the MNIST dataset in CSV format from https://www.kaggle.com/c/digit-recognizer/data (this is
just the training data, but enough for our purposes). Copy it to
  `data/mnist/train.csv`.
- Use `mnist-data-convert.pl` to split the dataset into the 8-class training
and test set, as well as the "new" set containing the remaining 2 digits,
and convert the data to Lua scripts.
- Run `TrainCluster.lua` to train the network and then use it to embed the
"new" images in 3D, and display it.
