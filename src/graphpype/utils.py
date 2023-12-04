import sys, os, numpy
from graphpype import pipe

def loadParcellation(*totalDataset, dataDirectory="", listFilterDirectory="", channel="", nameFilter=False):
    "Loading a combined dataset which has a specific parcellation and is filtered into subdatasets according to a filter list. If listFilter is empty then all parcellations are loaded into the data field of the combined dataset; else load each parcellation into the data field of the indexed data entry in the combined dataset indexed by name if provided else by order. Expects a matrix of floats, a (possibly empty) list of strings, and a dataset."
    mat = numpy.load(dataDirectory)
    if listFilterDirectory != "":
        listFilter = numpy.loadtxt(listFilterDirectory, dtype="str")
        labels = list(set(listFilter))
        indexed = [mat[listFilter == i, :] for i in labels]
        if nameFilter:
            for i in totalDataset:
                assert i.name in labels, "The filtering fields should correspond to the names in the dataset."
                idx = numpy.where([j == i.name for j in labels])[0][0]

                for j in range(numpy.shape(indexed[idx])[0]):
                    d = pipe.Datum({"string": "", "channel": "none"}) # intialise an empty datum
                    d.addChannel(channel, indexed[idx][j,:])
                    i.data.append(d)
        else:
            for i in range(len(indexed)):
                for j in range(totalDataset[i].shape[0]):
                    totalDataset[i].data.append(indexed[i][j,:])
    else:
        for i in range(mat.shape[0]):
            totalDataset.data.append(mat[i][:])
    return None

def loadAnalysisChannel(*totalDataset, dataDirectory="", channel="", dataType="csv"):
    """Loads data into a specified channel for all analysis groups. Very hacky."""
    match dataType:
        case "csv":
            data = numpy.genfromtxt(dataDirectory, delimiter=',')
        case "npy":
            data = numpy.load(dataDirectory)
        case "":
            data = open(dataDirectory)
    for i in totalDataset:
        i.analysis[channel] = data
    return None

def distanceMat(data):
    L = len(data)
    mat = numpy.zeros((L,L))
    for i in range(L):
        for j in range(L):
            mat[i,j] = numpy.sqrt(numpy.sum((data[i] - data[j]) ** 2))
    return mat

def vectorSlice(tensor, index, axis):
    slicei = tensor.take(indices=index, axis=axis)
    return slicei.reshape((numpy.size(slicei),))


def plots(*data, plotsDir):
    """Takes a plotting script and converts it into an operator."""

    sys.path.append(os.path.join(sys.path[0], plotsDir))
    import plots as p

    module_dictionary = p.__dict__

    res = []
    names = []

    for i in module_dictionary:
        f = module_dictionary[i]
        if callable(f):
            res.append(f(*data))
            names.append(i) 


    return res, names
