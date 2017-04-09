# TrainableOnlineClustering

A trainable clustering layer for neural networks suitable for online clustering

## Background

We aim to train a system that is capable of clustering items in a
semi-supervised online (streaming) setting where the number of clusters
is not known.
In other words, we have an initial, labelled batch of data, but after the initial
training the system should be capable of clustering / labelling a stream of
items, and recognise if new items form a new and unseen cluster (label).

The proposed solution is to use a regular neural network (ReLUs, LSTMs,
etc.) to embed
the items in a lower dimensional space (do dimension reduction from
a high number of features to a point in a low-dimensional Euclidean space),
and use a trainable clustering layer, implemented in this repository,
to cluster the resulting points.

The embedding network and the clustering layer are trained together
on the labelled data.
Afterwards, the clustering layer is detached and classic clustering methods
are used on the output of the embedding network.
With the right choice of clustering method this should allow the system to
recognise if items in the stream start forming a new hitherto unseen
cluster.


## The Clustering Layer

    D = number of dimensions
    O = number of labels / clusters (output)
    B = batch size

    O * B (probability of belonging to a cluster)
    
    ^ ^ ^ ^
    | | | |
    +-+-+-+----------------------------------+
    | Parameters:                            |
    | c[1,1], ..., c[O,D] (cluster midpoint) |
    | r[1], ..., r[O] (cluster radius)       |
    +-+-+-+----------------------------------+
    ^ ^ ^ ^
    | | | |
    
    D * B (item locations - real values)
   
For a given cluster, let r be the radius and \bar{c} = [c_1, c_2, ..., c_D] be the D-long vector
describing its center. Let \bar{x} = [x_1, x_2, ..., x_D] be a point in the low-dimensional place.
Then for the probability of this point belonging to the given cluster we use

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;P=e^{-\frac{|\bar{x}-\bar{c}|^2}{r^2}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;P=e^{-\frac{|\bar{x}-\bar{c}|^2}{r^2}}" title="P=e^{-\frac{|\bar{x}-\bar{c}|^2}{r^2}}" /></a>

We know that

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\frac{\partial&space;}{\partial&space;c_1}|\bar{x}-\bar{c}|&space;=&space;\frac{c_1-x_1}{|\bar{x}-\bar{c}|}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\frac{\partial&space;}{\partial&space;c_1}|\bar{x}-\bar{c}|&space;=&space;\frac{c_1-x_1}{|\bar{x}-\bar{c}|}" title="\frac{\partial }{\partial c_1}|\bar{x}-\bar{c}| = \frac{c_1-x_1}{|\bar{x}-\bar{c}|}" /></a>

and so

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\frac{\partial&space;P}{\partial&space;c_1}&space;=&space;\frac{\partial&space;P}{\partial&space;|\bar{x}-\bar{c}|}\frac{\partial&space;|\bar{x}-\bar{c}|}{\partial&space;c_1}&space;=&space;\frac{-2|\bar{x}-\bar{c}|}{r^2}P\frac{c_1-x_1}{|\bar{x}-\bar{c}|}=\frac{-2(c_1-x_1)}{r^2}P" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\frac{\partial&space;P}{\partial&space;c_1}&space;=&space;\frac{\partial&space;P}{\partial&space;|\bar{x}-\bar{c}|}\frac{\partial&space;|\bar{x}-\bar{c}|}{\partial&space;c_1}&space;=&space;\frac{-2|\bar{x}-\bar{c}|}{r^2}P\frac{c_1-x_1}{|\bar{x}-\bar{c}|}=\frac{-2(c_1-x_1)}{r^2}P" title="\frac{\partial P}{\partial c_1} = \frac{\partial P}{\partial |\bar{x}-\bar{c}|}\frac{\partial |\bar{x}-\bar{c}|}{\partial c_1} = \frac{-2|\bar{x}-\bar{c}|}{r^2}P\frac{c_1-x_1}{|\bar{x}-\bar{c}|}=\frac{-2(c_1-x_1)}{r^2}P" /></a>

and

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\frac{\partial&space;P}{\partial&space;r}&space;=&space;\frac{2|\bar{x}-\bar{c}|}{r^3}P" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\frac{\partial&space;P}{\partial&space;r}&space;=&space;\frac{2|\bar{x}-\bar{c}|}{r^3}P" title="\frac{\partial P}{\partial r} = \frac{2|\bar{x}-\bar{c}|}{r^3}P" /></a>
