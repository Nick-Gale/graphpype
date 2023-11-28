import networkx, scipy, numpy, statsmodels

def _squareMat(M):
    assert M.shape[0] == M.shape[1], "The provided matrix should be square."

def wiringCost(adj, dist):
    """Computes the standard wiring cost of a network defined by an adjancency matric and with a defined distance metric between the nodes. The cost (W) is the sum of the distances (D) weighted by the connectivity topology (A) normalised by the node number (N):
    W =1/N Σ AijDij 
    """
     
    sizeAdj = adj.shape
    sizeDist = dist.shape
    
    _squareMat(adj)
    _squareMat(dist)
    assert sizeAdj[0] == sizeDist[0], "The distance and adjacency matrix should have the same dimensions."

    return numpy.sum(datum.adjacency * datum.distance) / sizeAdj[0]

def louvainCommunities(G, seed: int, cuda=False):
    """A simple wrapper for the default parameters of the NetworkX implementation of the Louvain Clustering algorithm. The graph is weighted by the adjacency matrix and the seed must by specified according to the recipe."""

    # need to add support for GPU acceleration through cugraph
    
    return networkx.community.louvain_communities(G, seed=seed)

def greedyModules(G):
    """A simple wrapper for a greedy community detection algorithm using modularity maximisation."""
    # need to add support for GPU acceleration through cugraph

    if all((A == 0) ^ (A == 1)): # check adajancey is exclusively binary
        return networkx.greedy_modularity_communities(G)
    else:
        weights = networkx.get_edge_attributes(G, 'weight')
        return networkx.greedy_modularity_communities(G, weights=weights)

def degreeCDF(G):
    """Calculate the degree distribution as an empirical CDF. Returns a list of degree size and cummlative logged or unlogged CDF values (default: logProb = true). Finally, a model of the ecdf is returned from `statsmodels`."""
    
    degs = sorted([d for n, d in G.degree()], reverse=True)
   
    import statsmodels.api as sm # there must be a neater way to do this

    ecdf = sm.distributions.empirical_distribution.ECDF(degs)

    uniqueDegs = sorted(set(degs))

    cdf = numpy.array([ecdf(i) for i in uniqueDegs])

    logcdf = numpy.log(cdf)
    
    return uniqueDegs, cdf, logcdf, ecdf, degs

    
def degreeHistogram(G, nBins=1):
    """Calculate the degree histogram according to a particular number of bins (default: 1)."""
    
    degs = numpy.array(sorted([d for n, d in G.degree()], reverse=True))

    return numpy.histogram(degs, bins=nBins)

def featureDegreeDistribution(G, feature):
    """Concatenates features with the same degree and returns the mean and deviation"""
    degs = numpy.array([d for n, d in G.degree()])
    nodes = numpy.array([n for n, d in G.degree()])
    uniqueDegs = sorted(set(degs))
    
    featureData = []
    featureMean = []
    featureStandardDeviation = []
    for i in uniqueDegs:
        idxs = (i == degs)
        data = numpy.array(feature[nodes[idxs]].flatten())
        featureData.append(data)
        featureMean.append(numpy.mean(data))
        featureStandardDeviation.append(numpy.std(data))

    return uniqueDegs, featureData, featureMean, featureStandardDeviation

def euclideanDistanceMatrix(data, normalise=True):
    """Returns a distance matrix between parcellations. Expects the coordinates to be a vector for each parcellation and returns the Euclidean distances for each of them."""

    L = len(data[0]) # the dimension of the parcellation
    
    assert type(data[0][0]) == int, "The distances for all data should be in register: inumpy.t intepreted as multiple data points each with vectors associated with each parcellation location. Expects a single vector at each parcellation location"""

    dataMat = numpy.array(data)
    mat = numpy.zeros(L, L)
    for i in range(L):
        for j in range(L):
            v1 = utils.vectorSlice(datamat, i, 1)
            v2 = utils.vectorSlice(datamat, j, 1)
            mat[i,j] = numpy.sqrt(numpy.square(v1) + numpy.square(v2))
    return mat


def constructMinSpanDensity(covariance, density=0.1, seed=0):
    """ A covariance matrix (M) is used as a fully connected graph from which a minimal spanning tree (S) is constructed. The number of edges (N) in this spanning tree is calculated and the covariances asscociated with the nodes in the tree are set to zero. Finally, the difference (D) between the required density of edges (Nr) and the number of edges is calculated and D samples without replacement are taken using the remaining covariances as weights. These samples are added as edges into S to construct the final graph G."""

    numpy.random.seed(seed)
    
    L = covariance.shape[0]

    covariance = covariance - numpy.min(covariance) # CHECK THIS
    
    # construct the base graph using a minimal spanning tree
    M = networkx.Graph(covariance)

    Smat = networkx.to_numpy_array(networkx.minimum_spanning_tree(M))
    
    Smat[Smat > 0] = 1
    # sample required edges using covariances as weights
    nedges = numpy.sum(Smat)

    edges_required = int(numpy.round(L * (L-1) * density - nedges))

    unsampled_covariances = covariance * (1 - Smat)

    weights = unsampled_covariances.reshape(-1) / numpy.sum(unsampled_covariances)
    
    ascending_order = numpy.sort(weights)
    minimum_weight = ascending_order[-edges_required]
    
    edge_index_sample = weights > minimum_weight

    # construct edges as adjacency matrix
    edges = numpy.zeros(len(weights))

    edges[edge_index_sample] = 1 
    
    adj = Smat + edges.reshape(L, L)

    return networkx.DiGraph(adj)

def constructedDensityPermutationGraph(covariances, density=0.1, nPermutations=1000, seed=0):

    numpy.random.seed(seed)
    
    L = covariances.shape[0]
    
    permutations = [numpy.random.permutation(L) for i in range(nPermutations)]

    distribution = []
    
    for i in range(nPermutations):
        perm = numpy.random.permutation(L)
        permDict = dict(zip(range(L), perm))
        
        new_covariance = covariances.copy()
        new_covariance = new_covariance[perm, :]
        new_covariance = new_covariance[:, perm]
        
        g = constructMinSpanDensity(new_covariance, density=density, seed=seed)

        g = networkx.relabel_nodes(g, permDict)

        distribution.append(g)

    return distribution

def randomCommunityStochasticBlock(g, communities, density=0.1, nGraphs=1000, seed=0):
    """Given a graph G with an already computed community structure use the stochastic block model to create random distribution of graphs with similar community structure. The probabilities are nominally given as a representative edge denisty of the constructed network"""
    sz = [len(i) for i in communities]
    p = density * numpy.ones((len(sz), len(sz)))
    graphs =  [networkx.generators.community.stochastic_block_model(sz, p, seed=(seed + i)) for i in range(nGraphs)] 
    communities = [networkx.algorithms.community.louvain.louvain_communities(i, seed=seed) for i in graphs]
    return communities
