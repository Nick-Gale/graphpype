import networkx, scipy, numpy, statsmodels, nilean
from graphpype import utils

def anatNifti(data, atlas="msdl", atlasDir="./data/atlases", standardize="zscore_sample", standardize_confounds="zscore_sample", memory="nilearn_cache", verbose="0", confounds=False):
    "A simple API wrapper for the nilearn function for constructing a covariance matrix according to an atlas from Nifti data formats."
    
    atlasObj = utils.fetchAtlas(atlas, atlasDir)
    
    maps = atlasObj[0]
    # convenience function to map the regions of interest
    masker = NiftiMapsMasker(maps_img=maps, standardize=standardize, standardize_confounds=standize_confounds, memory=memory, verbose=verbose,)

    # generate the covariance matrix
    from sklearn.covariance import GraphicalLassoCV
    estimator = GraphicalLassoCV()
    estimator.fit(time_series)
    return estimator._covariance

def covNifti(data, atlas="msdl", atlasDir="./data/atlases", standardize="zscore_sample", standardize_confounds="zscore_sample", memory="nilearn_cache", verbose="0", confounds=False):
    "A simple API wrapper for the nilearn function for constructing a covariance matrix according to an atlas from Nifti data formats."
    
    atlasObj = utils.fetchAtlas(atlas, atlasDir)
    
    maps = atlasObj[0]
    # convenience function to map the regions of interest
    masker = NiftiMapsMasker(maps_img=maps, standardize=standardize, standardize_confounds=standize_confounds, memory=memory, verbose=verbose,)

    # generate the time series according to the regions of interest provided in the atlas
    if confounds:
        # confounds are located in the second index of the data
        time_series = masker.fit_transform(data[0], confounds=data[1])
    else:
        time_series = masker.fit_transform(data)

    # generate the covariance matrix
    from sklearn.covariance import GraphicalLassoCV
    estimator = GraphicalLassoCV()
    estimator.fit(time_series)
    return estimator._covariance
    
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
    res = networkx.algorithms.community.louvain.louvain_communities(G, seed=seed) #  networkx.community.louvain_communities(G, seed=seed)
    return res

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

def graphComposite(datum, features):
    """Takes a datum and a series of features to compose a graph composite. This could include multiple graphs each with multiple features on both edges and nodes. The graph composite is ideal for constructing tensorflowGNN graphs and for specifying regression/classification taasks on multiple graph data."""
    
    import tensorflow
    # features is a dictionary specifying the graph channel, the edge feature channels, and the node feature channels
    graphData = getattr(d, features["graph"])
    edges = graphData.edges()
    nodes = graphData.nodes()
    
    nodeFeatures = tensorflow.constant([getattr(d,f) for f in features["nodes"]])
    assert all([len(nF) == len(nodes) for nF in nodeFeatures]), "The node features must map onto the nodes"
    
    edgeFeatures = [getattr(d, f) for f in features["edges"]]
    assert all([len(eF) == len(edges) for eF in edgeFeatures]) or all([eF.size == [len(nodes), len(nodes)]), "The edge features must either map onto the nodes or be provided as an adjacency matrix."
    
    # edge notation
    for eF in edgeFeatures:
        if eF.size == [len(nodes), len(nodes)]:
            eF = [eF[e[0], e[1]], for e in edges]

    nodeSet = {
        "sizes": tensorflow.constant(len(nodes)),
        "features": dict(zip(features["nodes"], nodeFeatures))
            } 
    
    # if there is more than adjacency data to be included (e.g. fibre thicfeaturesness between two sites considered to be connected)
    edgeSet = {
        "sizes": tensorflow.constant(len(edges))
        "adjacency": {"source": (features["graph"], tf.constant([e[0] for e in edges])), "target"=(features["graph"], tf.constant([e[1] for e in edges]))},
        "features" = {dict(zip(features["edges"], edgeFeatures))} 
            }

    return nodeSet, edgeSet

    


def randomCommunityStochasticBlock(g, communities, density=0.1, nGraphs=1000, seed=0):
    """Given a graph G with an already computed community structure use the stochastic block model to create random distribution of graphs with similar community structure. The probabilities are nominally given as a representative edge denisty of the constructed network"""
    sz = [len(i) for i in communities]
    p = density * numpy.ones((len(sz), len(sz)))
    graphs =  [networkx.generators.community.stochastic_block_model(sz, p, seed=(seed + i)) for i in range(nGraphs)] 
    communities = [networkx.algorithms.community.louvain.louvain_communities(i, seed=(seed + s)) for s, i in enumerate(graphs)]
    return communities

def randomSpin(atlas="msdl", atlasDir="./data/atlases/", nPermutations=1000, seed=0):
    """A native implementation of the method proposed by Alexander-Block et. al. (2018) to control for spatial autocorrelation. Given data that is spatially distributed with a notion of spherical symmetry and an atlas of distance coordinates a spherical rotation is applied to the 3D space of the data and the atlas region associated with each datum is remapped to the closest (as the crow flies) atlas region. The data is typically in the form of feature (or biomarker) and can be a 3D or 4D tensor corrospending to some (registered) measurement. Currently, an atlas must be provided and it is rotated to give a new rotated atlas.

    Usage notes: Spinning a set of coordinates and aligning them to an atlas is equivalent to reverse-spinning the atlas. The latter is more space efficient does not compound errors (outside of those in the original atlas mapping). A diveregence from the original implementation is the generation of random numbers; the original paper stated rotations on the sphere were uniformly distributed but the Git repository indicated sampling from a normal distribution and enforcing orthoganality via QR decomposition. This is a costly procedure. Here we will sample each angle from the distribution U([0,1]) and transform to the correct range for appropriate Euler angles.
    
    To do: offer acceleration through CUDA
    """

    # create the rotational distribution of Euler angles
    sample = numpy.random.rand(3, nPermutations)

    alphaPerm = 2 * numpy.pi * sample[1]

    betaPerm = numpy.pi * (sample[2] - 0.5)

    gammaPerm = 2 * numpy.pi * sample[3]
    
    angles = zip(alphaPerm, betaPerm, gammaPerm)
 
    # create the rotations
    
    leftRotationOperator = numpy.array([])
    rightRotationOperator = numpy.array([])
    leftRightTransform = numpy.array([[-1,0,0],[0,1,0],[0,0,1]])
    for eulerAngle in angles:
        cosAlpha, cosBeta, cosGamma = numpy.cos(eulerAngle)
        sinAlpha, sinBeta, sinGamma = nump.sin(eulerAngle)

        left = numpy.array([
            [cosBeta * cosGamma, sinAlpha * sinBeta * cosGamma - cosAlpha * sinGamma, cosAlpha * sinBeta * cosGamma + sinAlpha * sinGamma],
            [cosaBeta * sinGamma, sinAlpha * sinBeta * sinGamma + cosAlpha * cosGamma, cosAlpha * sinBeta * sinGamma - sinAlph * cosGamma],
            [-sinBeta, sinAlpha * cosBeta, cosAlpha *  cosBeta]
            ])
        right = left * leftRightTransform

        leftRotationOperator.append(left.tranpose()) # take the inverse of the left rotation as we are rotating the atlas not the data
        rightRotationOperator.append(right.tranpose()) # take the inverse of the right rotation

    # apply rotations; in Nibabel formats RAS+ space is assumed which means the central voxel in the first index corresponds to the left-right midline
    atlasObj = utils.fetchAtlas(atlas, atlasDir)
    
    # Let affine be A, rotation matrix be R, and voxel index tuple be v then: A^(-1) R A v = vr

    data = atlasObj.maps._dataobj
    A = atlasObj.maps._affine
    iA = numpy.inverse(A)

    voxInds = numpy.array( [numpy.unravel_index(i, data.shape) for i in range(data.size)])
    midPoint = data.shape[0] // 2 # integer division
    voxLeft = voxInds[voxInds[:,0] < midPoint]#numpy.take(voxInds, indices=arange(0,midPoint), axis)
    voxRight = voxInds[voxInds[:,0] >= midPoint]#numpy.take(voxInds, indices=arange(midPoint, data.shape[0]), axis=0)   
    voxList = list(map(tuple, voxLeft)).append(list(map(tuple, voxRight)))
    voxSet = set(voxList)
   
    permutations = []
    for (l, r) in zip(leftRotationOperator, rightRotationOperator):
        voxLeftTransformed = (iA * l * A * voxLeft).round()
        voxRightTransformed = (iA * r * A * voxRight).round()
        
        # there is a possibility that these are not unique!
        voxListTransformed = list(map(tuple, voxLeftTransformed)).append(list(map(tuple, voxRightTransformed)))
        vSet = set(voxListTransformed)
        required = voxSet - vSet
        if required:
            # loop through the required elements filling over the duplicated elements
            count = 0
            seen = set()
            for idx in range(len(voxListTransformed)):
                if voxListTransformed[x] in seen:
                    voxListTransformed[x] = required[count]
                    count += 1
                else:
                    seen.add(voxListTransformed[x])
        
        # create the permutation to remap data idx (a,b,c) -> (i,j,k)
        append.permutation(zip(voxList, voxListTransformed))

    # permute the atlas data; seems data inefficient should we just return the permutation?
    permutedAtlases = []
    for p in permutations:
        newAtlas = copy.deepcopy(atlasObj)
        m = copy.deepcopy(newAtlas.maps._dataobj)
        for (uT, T) in p: # destroys this permutation
            newAtlas.maps._dataobj[uT[0], uT[1], uT[2], ...] = m[T[0], T[1], T[2], ...]
        permutedAtlases.append(newAtlas)
    
    return permutedAtlases

