import scipy, numpy, statsmodels, neuromaps, networkx, warnings
from graphpype import utils
def estimateDistancePermutation(graph, distanceDist):

    dist = numpy.zeros(len(graph.edges()))
    for i, e in enumerate(graph.edges()):
        
        dist[i] = numpy.sqrt(numpy.sum((distanceDist[e[0]] - distanceDist[e[1]]) ** 2))
    av = scipy.mean(dist)
    err = scipy.stats.sem(dist)
    return dist, av, err

def modularOverlap(modules1, modules2):
    """Given a vector of module membership of nodes compute the modular overlap and return a z-transformed score. To compute the modular overlap compute the fraction of pairs of nodes that share a module in both groups i.e. a binary vector. Note: this is not a symetric relationship between partitions as the vectors will have different lengths based on which is chosen first. Return the mean/std of the vector."""
    modules = [[sorted(k) for k in i] for i in [modules1, modules2]]
    statistic = 0 
    for modi in modules[0]:
        for modj in modules[1]:
            N = len(numpy.intersect1d(modi, modj)) # nodes in modi that share a module classification in the second graph
            npairs = N * (N - 1) / 2 # number of pairs that can be made with these nodes
            statistic += npairs
    nNodes = max([max(k) for k in modules[0]])
    totalPairs = nNodes * (nNodes - 1) / 2
    return statistic / totalPairs

def modularZTest(*modules):
    """ """
    L = len(modules)
    zstats = numpy.zeros((L, L))
    for i in range(L):
        dist = [modularOverlap(modules[i][0], k) for k in modules[i][1]]
        for j in range(L):
            if i == j:
                zstats[i, j] = -1
            else:
                measured = modularOverlap(modules[i][0], modules[j][0])
                zstats[i,j] = abs(numpy.mean(dist) - measured) / numpy.std(dist)
    return zstats

def pairgroupModularZTest(*modules, correction="FDR", threshold=0.025):
    """Compute the modular overlap of each of the measured populations and each paired sample from the null model. The z-test then computes Z values for each paired difference of the modular overlap of each pair of groupings when compared against the paired difference of distributions in each pair of groupings. These p-values are corrected (default: FDR) and returned as a matrix of pairs of the paired groupings and the signficance values are reported. """
    
    L = len(modules)
    N = [len(modules[i][1]) for i in range(len(modules))]
    measured = numpy.zeros(L*L)
    measuredDist = numpy.zeros((L*L, max(N)))
    
    for i in range(L):
        for j in range(L):
            measured[i*L + j] = modularOverlap(modules[i][0], modules[j][0])
            measuredDist[i*L + j, 0:N[i]] = numpy.array([modularOverlap(modules[i][1][k], modules[j][1][k]) for k in range(N[i])]) 
            
    pairedPval = numpy.zeros((L*L, L*L))
    for p in range(L*L):
        for q in range(L*L):
            pairedStat = measured[p] - measured[q]
            pairedNull = measuredDist[p, :] - measuredDist[q, :]
            
            if all(pairedNull == 0):
                warnings.warn("No modular overlap", Warning)
                pairedNull += 10 ** (-14)

            pairedZ = (numpy.mean(pairedNull) - pairedStat) / numpy.std(pairedNull)
            pairedPval[p, q] = scipy.stats.norm.sf(abs(pairedZ)) * 2

    pval = pairedPval.flatten()
    pval[numpy.isnan(pval)] = 1 
    match correction:
        case "FDR":
            pval = scipy.stats.false_discovery_control(pval)
            sig = pval < threshold
        case "none":
            sig = pval < threshold
        case "Bonferroni":
            pval = pval * (L * L)
            sig = pval < threshold

    return pval.reshape(L*L, L*L), sig.reshape(L*L, L*L)

def compareGroupDegreeMeans(*data, channel="", correction="FDR", threshold=0.05):
    """Returns the pairwise t-test between a list of data for each degree in a groupwise graph. These tests are corrected and significant degrees are reported."""
    # This function isn't particularly elegant
    M = max([max([len(i) for i in d]) for d in data])
    

    padded_data = numpy.array([[numpy.concatenate([numpy.zeros(M - len(j)), j]) for j in d] for d in data])
    groups = [numpy.array(d) for d in padded_data]
    names = list(range(len(data)))

    means = [numpy.mean(g, axis=0) for g in groups]
    standard_error = [numpy.std(g, axis=0) / numpy.sqrt(numpy.shape(g)[0]) for g in groups]
    
    res = {}
    for i, name in enumerate(names):
        res[names[i]] = {"mean": means[i], "error": standard_error[i]}
    
    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            if i == j:
                pvals = numpy.ones(len(means[i]))
            else:
                pvals = numpy.zeros(len(means[i]))
                for k in range(len(means[i])):
                    # check that the distributions aren't identical, else give a pval of 1
                    if all(groups[i][:, k] == groups[j][:, k]):
                        pvals[k] = 1
                    else:
                        pvals[k] = scipy.stats.ttest_ind(groups[i][:,k], groups[j][:, k]).pvalue
                
            nans = numpy.isnan(pvals)
            if any(nans):
                warnings.warn("You have NaNs in your pvalues", Warning)
                pvals[nans] = 1

            match correction:
                case "FDR":
                    scipy.stats.false_discovery_control(pvals)
                    res[name1][name2] = {"pvalues": pvals, "mask": pvals < threshold}
                case "none":
                    res[name1][name2] = {"pvalues": pvals, "mask": pvals < threshold}
    
    return res

def compareDist(dist1, dist2, test='Kolmogorov-Smirnov'):
    """Assesses the degree to which two empirical distributions are statistically different. There are some common ways of doing this: assuming the degree distribution takes a specific functional form, or the general Kolmogorov-Smirnov test. Support is provided for degree distributions assumed to be in the power-law form or the Kolmogorov-Smirnov through the `test` keyword  (default `test='Kolmogorov-Smirnov'`)"""

    # assert type is list of integers, or two ecdfs
    assert (type(dist1) == list or type(dist1) == numpy.ndarray or type(dist1) == statsmodels.distributions.emperical_distribution), "The first distribution should have a type: list, numpy array, or ecdf"
    assert (type(dist2) == list or type(dist2) == numpy.ndarray or type(dist2) == statsmodels.distributions.emperical_distribution), "The second distribution should have a type: list, numpy array, or ecdf"
    
    match test:
    # case switch KS or powerlaw
        case 'Kolmogorov-Smirnov':
            return scipy.stats.ktest(dist1, dist2)


def generatePermutations(datum, n, method='degree', seed=0):
    """Generate a random set of permutations for a given adjacency matrix that serve as a null-model distribution under some assumption i.e. preserving degree distribution. Currently supported options are: 'degree', 'spin', 'random'. Degree distributions are generated using the FILLTHISIN method from FILLTHISINPACKAGE/PAPER. Spin models follow the structure presented by Block et. al. and implemented in FILLTHISINPACKAGE. Random assumes no structure and provides random permutations of the matrix."""
    
    A = datum.adjacency
    L = A.shape[0] 

    match method:
        case 'degree':
            ds = numpy.array(sorted([d for n, d in datum.graph.degree()], reverse=True))
            adjs = [networkx.configuration_model(ds, seed=(seed+i)).adjacency.todense() for i in range(n)]
        case 'spin':
            adjs = []
    return adjs

def covarianceMatrix(*data, normalise=True):
    """Returns the covariance matrix (size: L x L) of a data distributed amongst a particular parcellation (size: L) for a particular dataset (size: N). The default behavior is to normalise by standard deviations returning the regular Pearsons correlation coefficient."""
    dims = numpy.shape(data[0][0]) # L the dimension of the parcellation, y the dimension of the data e.g. single estimate or a time series
    if len(dims) == 1:
        L = dims[0]
        y = 1
    else:
        L = dims[0]
        y = dims[1]

    assert all([len(data[0][0]) == len(data[i][0]) for i in range(len(data))]), "All covariance comparisons should be of the same data dimensionality."
    
    if y != 1:
        dataMat = numpy.array(data)
    else:
        dataMat = numpy.array([i[0] for i in data])
        dataMat = dataMat.transpose()
    
    mat = numpy.zeros((L, L))
    
    if y == 1:
        covmat = numpy.cov(dataMat)
        if normalise:
            std = numpy.std(dataMat, axis=1, ddof=1)
            covmat /= numpy.outer(std, std)

    for i in range(L):
        for j in range(L):
            if i ==j:
                mat[i,j] = 0
            else:
                if y == 1:
                    mat[i,j] = covmat[i,j]
                else:
                    vi = utils.vectorSlice(dataMat, i, 2)
                    vj = utils.vectorSlice(dataMat, j, 2)
                    if normalise:
                        mat[i, j] = scipy.stats.pearsonr(vi, vj).statistic
                    else:
                        mat[i, j] = numpy.cov(vi, vj)[0][1]

    return mat

def loadFeature(*data):

    data = [i[0] for i in data] # not sure about this
    assert len(numpy.shape(data[0])) == 1

    dataMat = numpy.array(data).transpose()
    
    return dataMat

def multipleTTest(*data, threshold=0.025, correction="FDR"):

    ndata = len(data)
    
    names = range(ndata)
    
    nres = (ndata - 1) * ndata / 2

    res = {i: dict() for i in names}

    for i in range(ndata-1):
        for j in range(i+1, ndata):
            pvals = []
            idx = []
            for k, deg in enumerate(data[i][0]):
                if deg in data[j][0]:
                    idx.append(deg)
                    dataI = numpy.array(data[i][1][k])
                    idJ = numpy.where(data[j][0] == deg)[0][0]
                    dataJ = data[j][1][idJ]
                    pvals.append(scipy.stats.ttest_ind(dataI, dataJ).pvalue)

            match correction:
                case "FDR":
                    pvals = scipy.stats.false_discovery_control(pvals)
                    sig = pvals < threshold
                case "Bonferroni":
                    pvals = pvals * length(pvals)
                    sig = pvals < threshold
                case "none":
                    sig = pvals < threshold
                
            res[i][j] = {"pvals": pvals, "significance": sig, "idxs": idx} 
            
    return res


def generalLinearModel(*data, sets=[], covariateChannels=[], regressorChannels=[], link=None, test='none', flatten=True):
    
    if sets == []:
        # regressing on a single data set
        if flatten:
            fitData = [numpy.array(d).flatten() for d in data]
        assert len(fitData) == 2, "There should only only be two arrays: covariates and regressors"
        
        assert numpy.shape(fitData[0])[0] == numpy.shape(fitData[1])[0], "The covariates and regressors should have the same number of data points."
        import statsmodels.api as sm 
        return sm.OLS(fitData[0], statsmodels.tools.tools.add_constant(fitData[1])).fit()

    else:
        # regressing on multiple datasets
        fit = {}     
        for x in range(len(sets)):
            fit[x] = {}
            for y in range(len(sets)):
                if y != x:
                     
                    # Load data by index    
                    if type(sets[x]) == str:
                        index = where([sets[x] == i.name for i in data.data])
                        dataX = data.data[index]
                    elif type(sets[x]) == int:
                        dataX = data.data[x]
                    else:
                        raise("Please address the data by a named index, or an integer corresponding to the linear index of the dataset")

                    if type(sets[y]) == str:
                        index = where([sets[y] == i.name for i in data.data])
                        dataY = data.data[index]
                    elif type(sets[x]) == int:
                        dataY = data.data[y]
                    else:
                        raise("Please address the data by a named index, or an integer corresponding to the linear index of the dataset")
                    
                    regressors = numpy.array([getattr(d, c) for c in regressorChannels for d in dataX])
                    covariates = numpy.array([getattr(d, c) for c in covariateChannels for d in dataY])
                    if link==None:    
                        glm = statsmodels.OLS(regressors, covariates).fit()
                    else:
                        linkfamily = getattr(statsmodels.families, link)
                        glm = statsmodels.GLM(regressors, covariates, family=linkfamily).fit()
                    
                    fit[x][y]["model"] = glm
        return fit

def graphNeuralNetwork(data, graphComposites=[], network={}, learningTask={}):
    """Generalised graph neural networks API call to abstract arbitrary graph data formats and train them following the tfgnn GraphTensor structure.

    Usage:

    The data is treated as a dataset and a dictionary of "graphs" are passed per subject and the node and edge sets in the GraphTensor. Each graph is attached to a keyword e.g. "fmri" or "adjacency" and each of these can hold feature lists for every graph in the data in both the edges and the nodes. Currently, these graphs are considered to be independent i.e. there are assumed to be no edges between each keyword although in principle there is nothing stopping links between, for example, gene regulatory networks and fmri images. The data is specified as a total dataset at either the analysis or dataset level. 

    Graph composites are in the form of a dictionary and specify the channel of the graph, the channels where the node features are derived, and the channels where the edge features are derived. These are used to derive the subgraphs and features extracted from each datum in the dataset to combine into the final graphTensor representing the entire dataset.

    `network` specifies the architecture of the graph neural network i.e. the message parsing format and the number of message passes to make. The final layer must be a dense network to the feature size of the learning context.

    `learningTask` specifies the learning task. It is composed of a dictionary that specifies the training split (by dataset, or by percentage), the optimiser, the epochs, the batch, etc. Example: learningTask = {task: {name: "name", parameters: params...}, batchSize=32, nEpochs: 5, validationsPerEpoch: 2, optimizer: "Adam"}
:w

    The output is a trained keras model which can be used for inference.
    """
    import tensorflow
    import tensforlow_gnn as tfgnn

    # construct the graph data in tensorflow objects
    graphDataSets = []
    for dataSet in data:
        for d in dataSet:
            nodeSets = {}
            edgeSets = {}
            context = {}
            for gf in graphComposites: # number of components
                gC = graphComposite(d, gf)
                
                graphName = gf["graph"]
                
                nodeSets[graphName]["sizes"].append(gC[0]["sizes"])
                nodeSets[graphName]["features"].append(gC[0]["features"])

                edgeSets[graphName]["sizes"].append(gC[1]["sizes"])
                edgeSets[graphName]["adjacency"]["source"].append(gC[1]["source"][1])
                edgeSets[graphName]["adjacency"]["target"].append(gC[1]["target"][1])
                edgeSets[graphName]["features"].append(gC[1]["features"])
                
                
                # ADD THE CONTEXTT HERE[

            # make the nodeSets and edgeSets compatible with tfgnn
            for (graphName, n) in nodeSets:
                n = tfgnn.NodeSet.from_fields(sizes=n["sizes"], features=n["features"])

            for (graphName, n) in edgeSets:
                n["adjacency"] = tfgnn.Adjacency.from_indices(source=(graphName, n["adjacency"]["source"]), target=(graphName, n["adjacency"]["target"]))
                n = tfgnn.EdgeSet.from_fields(sizes=n["sizes"], adjacency=n["adjacency"], features={}) # TO DO: adapt features

            
            dTensorFlowGraph = tfgnn.GraphTensor.from_pieces(
                    context=tfgnn.Context.from_fields(features=dGraphContext), # the learning context
                    nodeset=nodeSets, # the features attached to each node of each subgraph that compose the graph composite e.g. fMRI atlas + gene reg network
                    edgeset=edgeSets # the features attached to the edges of each node of each subgraph permitting subgraph - subgraph nodes. This is not typical.
                )
            graphDataSets.append(dTensorFlowGraph) # perhaps better to squash datasets together and remeber the indexes for test/train split
    
    if len(data) == 1:
        assert 0 < learningTask["trainingSplit"]) < 1, "If a single dataset is selected you must specify the test-training split as a float in (0,1)"
        trainData = graphDataSets[0:int(round(learningTask["trainingSplit"]))]
        testData = graphDataSets[int(round(learningTask["trainingSplit"])):]
    elif len(data) == 2:
        trainData = graphDataSets[0:len(data[0])]
        testData = graphDataSets[len(data[0]):]
    else:
        raise("The training data must be composed correctly; either as a split dataset indicated by learningTask[:trainingSplit] or as two datasets")
   
    # dump the data
    zipped = zip([trainData, testData], ["train", "test"])
    for (dataSet, dtype) in zipped:
        path = learningTask["bidsPath"] + '_' + dtype + '.tfrecord'
        with tf.io.TFRecordWriter(filename) as writer:
            for graph_tensor in dataSet: 
                example = tfgnn.write_example(graph_tensor)
                writer.write(example.SerializeToString())



    # create the datasets TFGNN spec
    trainDataSetProvider = tfgnn.runner.TFRecordDatasetProvider(file_pattern=learningTask["bidsPath"] + '_train.tfrecord')
    trainDataSetProvider = train_dataset_provider.get_dataset(context=tf.distribute.InputContext())
    trainDataSetProvider = trainDataSet.map(lambda serialized: tfgnn.parse_single_example(serialized=serialized, spec=graphSpec))

    testDataSetProvider = tfgnn.runner.TFRecordDatasetProvider(file_pattern=learningTask["bidsPath"] + '_validate.tfrecord')
    testDataSetProvider = train_dataset_provider.get_dataset(context=tf.distribute.InputContext())
    testDataSetProvider = testDataSet.map(lambda serialized: tfgnn.parse_single_example(serialized=serialized, spec=graphSpec))
    
    # construct the neural network
    def modelFunction(graphSpec)
        # The meat of the graph neural network model: currently restricted to "existing" architectures. TO DO: add generic implementation features outside of the "define your own function" functionalitly provided by graphpype
        graph = inputGraph = tf.keras.layers.Input(type_spec=graphSpec)
        # implement the message passing function from available API calls; error messages fallback on the tfgnn errors
        modelAPI = getattr(tfgnn.models, network["model"]["name"])
        updateFunction = getattr(modelAPI, "GraphUpdate")
        for n in range(learningTask["messagePassingUpdates"]):
            # conduct several rounds of message parsing and create trainable layers for each of these
            graph = updateFunction(network["model"]["parameters"])(graph) # the parameters are specified as a dictionary corresponding to the docs for each model api 
        return tensorflow.keras.Model(inputGraph, graph)
    
    # set the initial node states according to a lambda function which extracts over the node name
    mapFeatures = tfgnn.keras.layers.MapFeatures(node_sets_fn=network["initialNodeStatesFunction"])

    # define the task (i.e. node classification) according to the API implemenations provided by TensorFlow. This has the same problems as neural networks above.
    taskAPI = getattr(tfgnn.runner, learningTask["task"]["name"])
    task = taskAPI(learningTask["task"]["parameters"])

    trainer = tfgnn.runner.KerasTrainer(
            strategy=tf.distribute.TPUStrategy(...),
            model_dir="...",
            steps_per_epoch=len(trainData) // learningTask["batchSize"],
            validation_per_epoch=learningTask["validationsPerEpoch"],
            steps_per_validation=len(testData) // learningTask["batchSize"]
            )

    # train
    res = tfgnn.runner.run(
                train_ds_provider=trainDataProvider,
                train_padding=runner.FitOrSkipPadding(graphSpec, trainDataProvider),
                model_fn=model_fn,
                optimizer_fn=tf.keras.optimizers.Adam,
                epochs=learningTask["nEpochs"],
                trainer=trainer,
                task=task,
                gtspec=graphSpec,
                global_batch_size=learningTask["batchSize"],
                feature_processors=[mapFeatures],
                valid_ds_provider=testDataSetProvider
                )

    return res
            
                
            
