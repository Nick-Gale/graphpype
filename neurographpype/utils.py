import sys, os, numpy

def loadParcellation(directory, listFilter=str[], totalDataset, nameFilter=False):
    "Loading a combined dataset which has a specific parcellation and is filtered into subdatasets according to a filter list. If listFilter is empty then all parcellations are loaded into the data field of the combined dataset; else load each parcellation into the data field of the indexed data entry in the combined dataset indexed by name if provided else by order. Expects a matrix of floats, a (possibly empty) list of strings, and a dataset."

    mat = numpy.load(directory)

    if listFilter:
        labels = set(listFilter)
        indexed = [listFilter[listFilter == i, :] for i in labels]
        filt = zip(labels, indexed)
        if nameFilter:
            for i in totalDataset.data:
                assert i.name in filt, "The filtering fields should correspond to the names in the dataset."
                for j in range((filt[i.name].shape[0]):
                    i.data.append(filt[i.name][j,:])
        else:
            for i in range(len(indexed)):
                for j in range(i.shape[0]):
                    totalDataset.data[i].data.append(indexed[i][j,:])
    else:
        for i in range(mat.shape[0]):
            totalDataset.data.append(mat[i][:])
    return None

def vectorSlice(tensor, index, axis):
    slicei = tensor.take(indices=index, axis=axis)
    return slicei.reshape((numpy.size(slicei),))
