import networkx, scipy, numpy

def _squareMat(M)
    assert M.shape[0] == M.shape[1], "The provided matrix should be square."

def wiringCost(adj, dist):
    """Computes the standard wiring cost of a network defined by an adjancency matric and with a defined distance metric between the nodes. The cost (W) is the sum of the distances (D) weighted by the connectivity topology (A) normalised by the node number (N):
    W =1/N Σ AijDij 
    """
     
    sizeAdj = datum.adjacency.shape
    sizeDist = datum.distance.shape
    
    _squareMat(adj)
    _squareMat(dist)
    assert sizeAdj[0] == sizeDist[0], "The distance and adjacency matrix should have the same dimensions."

    return numpy.sum(datum.adjacency * datum.distance) / sizeAdj[0]

def louvainCommunities(datum, seed: int, cuda=False):
    """A simple wrapper for the default parameters of the NetworkX implementation of the Louvain Clustering algorithm. The graph is weighted by the adjacency matrix and the seed must by specified according to the recipe."""

    # need to add support for GPU acceleration through cugraph

    G = networkx.DiGraph(datum.adjacency)

    if all((A == 0) ^ (A == 1)): # check adajancey is exclusively binary
        return networkx.louvain_communities(G, seed=seed)
    else:
        weights = networkx.get_edge_attributes(G, 'weight')
        return networkx.louvain_communities(G, weights=weights, seed=seed)

def greedyModules(datum):
    """A simple wrapper for a greedy community detection algorithm using modularity maximisation."""
    # need to add support for GPU acceleration through cugraph
   
    G = networkx.DiGraph(datum.adjacency)

    if all((A == 0) ^ (A == 1)): # check adajancey is exclusively binary
        return networkx.greedy_modularity_communities(G)
    else:
        weights = networkx.get_edge_attributes(G, 'weight')
        return networkx.greedy_modularity_communities(G, weights=weights)

def degreeCDF(datum):
    """Calculate the degree distribution as an empirical CDF. Returns a list of degree size and cummlative logged or unlogged CDF values (default: logProb = true). Finally, a model of the ecdf is returned from `statsmodels`."""
    
    G = networkx.DiGraph(datum.adjacency)

    degs = sorted([d for n, d in G.degree()], reverse=True)

    ecdf = statsmodel.ECDF(degs)

    uniqueDegs = set(degs)

    cdf = numpy.array([ecdf(i) for i in uniqueDegs])

    logcdf = numpy.log(cdf)

    return degs, cdf, logcdf, ecdf

    
def degreeHistogram(datum, nBins=25):
    """Calculate the degree histogram according to a particular number of bins (default: 25)."""
    
    G = networkx.DiGraph(datum.adjacency)

    degs = numpy.array(sorted([d for n, d in G.degree()], reverse=True]))

    return numpy.histogram(degs, bins=nBins)


def euclideanDistanceMatrix(data, normalise=True):
    """Returns a distance matrix between parcellations. Expects the coordinates to be a vector for each parcellation and returns the Euclidean distances for each of them."""

    L = len(data[0]) # the dimension of the parcellation
    
    assert type(data[0][0]) == int, "The distances for all data should be in register: input intepreted as multiple data points each with vectors associated with each parcellation location. Expects a single vector at each parcellation location"""

    dataMat = numpy.array(data)
    mat = numpy.zeros(L, L)
    for i in range(L):
        for j in range(L):
            v1 = utils.vectorSlice(datamat, i, 1)
            v2 = utils.vectorSlice(datamat, j, 1)
            mat[i,j] = numpy.sqrt(numpy.square(v1) + numpy.square(v2))
    
