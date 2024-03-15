import sys, os, numpy, subprocess, pickle, inspect, shutil
from graphpype import pipe

def generateTemplate(name="generic", exampleFile="generic.py"):
    """Generates a fill-in-the-blanks template to work with. This is generally easier than remembering all the particular details about how operators (and in particular some common Operators) should be specified. Writes the recipe immediately to the name specified and the recipe exists in the python session and can be dynamically edited.

    Usage notes: if dynamically editing in Python remember to call pipe.write("someRecipeName.json")"""

    rootDIR = Path(__file__).parent.parent.parent
    exampleDIR = rootDIR + '/' + exampleFile
    currentDir = os.getcwd()

    shutil.copyfile(exampleDir, currentDir + '/' + name + '.py')
    
    exec(open('filename').read())

    pipe.write(name + ".json")
    
    return recipe

def fetchAtlas(atlas="msdl", atlasDir="./data/atlases/"):
    "Grabs an atlas using the NiLearn API. The default atlas used is the msdl atlas but this can be specified to work with any atlas available in the NiLearn database."
    # check the atlas exists
    if os.path.exists(atlasDir + atlas):
        atlasObj = loadObj(atlasDir + atlas)
    else
        # fetch the atlas and labels
        funcStr = "datasets.fetch_atlas_" + atlas
        funcFetch = getattr(nilearn, funcStr)
        atlasObj = funcFetch()
        # seperate the regions into spatially continuous blocks (no left-right hemisphere symettery for example)
        atlasObj = connected_label_regions(atlasObj)
        # atlasMaps = atlasObj["maps"]
        # atlasLabels = atlas["labels"]
        
        saveObj(atlasObj, atlasDir)

    return atlasObj

def fmriprep(directory, fmriprep=[], participant=[], cache=True, stringAdd=""):
    # modify directory to be BIDS and terminal compliant:
    if directory[-1] != '/':
        directory += '/'
    derivativesDirectory = directory + 'derivatives/'
    
    # find the participant labels
    subsDerivative = [ f.path[len(derivativesDirectory):] for f in os.scandir(derivativesDirectory) if f.is_dir() and f.path[len(derivativesDirectory):][0:4] == 'sub-' ]
    subs = [ f.path[len(directory):] for f in os.scandir(directory) if f.is_dir() and f.path[len(directory):][0:4] == 'sub-' ]
    
    # if no participants are labelled presume all participants are selected
    if participant == []:
        participant = [i for i in subs]
    
    # once there is a non empty specification of participants refine the subs paths
    if type(participant[0]) == int:
        subsPaths = [subs[i] for i in participant]
    else:
        subsPaths = participant
    
    # check the derivatives directory for participants that have already been processed
    if cache:
        subsProcess = [i for i in subsPaths if i not in subsDerivative]
    else:
        subsProcess = subsPaths
    
    # a non-empty subsProcess should be passed to fmriprep, else cache has been selected
    if subsProcess: 
        cmdStr = ["fmriprep-docker", directory, derivativesDirectory, "participant"] + fmriprep + ["--participant-label"] + subsPaths
        subprocess.run(cmdStr)
    else:
        print("Previously preprocessed participants have been detected in the BIDS/derivatives/ directory. These will be selected as the `cache` option is True. To reprocess these set `cache=False`")

    return None

def loadObject(directory):
    with open(directory, 'wb') as file:  # Overwrites any existing file.
        obj = pickle.load(file)
    return obj

def saveObject(obj, directory):
    with open(directory, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

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
