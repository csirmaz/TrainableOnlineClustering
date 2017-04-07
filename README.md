# TrainableOnlineClustering

A trainable clustering layer for neural networks suitable for online clustering

## Background

We aim to train a system that is capable of clustering items in a
semi-supervised online (streaming) setting where the number of clusters
is not known.
In other words, an initial batch of data is labelled, but after the initial
training the system should be capable of clustering / labelling a stream of
items, and recognise if new items form a new and unseen cluster or label.

The proposed solution is to use a regular neural network (ReLUs, LSTMs,
etc.) to embed
the items in a lower dimensional space (do dimension reduction from
a high number of features to a point in a low-dimensional Euclidean space),
and use a trainable clustering layer, implemented in this repository,
to cluster points in the low-dimensional space.

The embedding network and the clustering layer are trained together
on the labelled data.
Afterwards, the clutering layer is detached and classic clustering methods
are used on the output of the embedding network.
With the right choice of clustering method this allows the system to
recognise if items in the stream start forming a new hitherto unseen
cluster.


## The Clustering Layer

    D = number of dimensions
    O = number of labels / clusters (output)
    C = batch size

    O * C (probability of belonging to a cluster)
    
    ^ ^ ^ ^
    | | | |
    +-+-+-+-------------------------+
    | Parameters:                   |
    | L[1,1], ..., L[O,D] (location)|
    | S[1], ..., S[O] (spread)      |
    +-+-+-+-------------------------+
    ^ ^ ^ ^
    | | | |
    
    D * C (item locations - real values)
    