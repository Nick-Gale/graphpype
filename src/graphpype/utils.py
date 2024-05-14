"""
Utility Functions
=================

A collection of utility functions that allow graphpype to operate but are not specific to neuroimaging or graphpype.

"""

import sys, os, numpy, subprocess, pickle, inspect, shutil
from graphpype import pipe

def generateTemplate(name="generic", exampleFile="generic.py"):
    r"""Autogenerate a template recipe to work from.

    The recipe will automatically be written and there are several basic templates provided in the package. The template can be dynamically edited in Python using the package tools or the generated .json file can be edited by a text editor.

    Parameters
    ----------
    name : string
        Defaults to generic.

    Returns
    -------
    recipe : graphpype.pipe.recipe
        A recipe object used to construct an analysis pipeline.

    Notes
    -----
    Editing in a live Python session should be saved with a call to the pipe.write function

    See Also
    --------
    graphpype.pipe.recipe.write : The live method of saving a recipe template.
    """
    
    rootDIR = Path(__file__).parent.parent.parent
    exampleDIR = rootDIR + '/' + exampleFile
    currentDir = os.getcwd()

    shutil.copyfile(exampleDir, currentDir + '/' + name + '.py')
    
    exec(open('filename').read())

    pipe.write(name + ".json")
    
    return recipe

def fetchAtlas(atlas="msdl", atlasDir="./data/atlases/"):
    r"""Grabs an atlas using the NiLearn API. 

    The default atlas used is the 'msdl' atlas but this can be specified to work with any atlas available in the NiLearn database. The atlases are placed in the `data/atlases/` subdirectory of the BIDS directory.

    Parameters
    ----------
    atlas : string
        Default is 'msdl' but can be any in the NiLean database e.g. 'cort-maxprob-thr25-2mm' for the Harvard-Oxford atlas.
    atlasDir: string
        Defaults to data/atlases in the BIDSs directory. This is BIDS compliant but can be changed.

    Returns
    -------
    atlasObj : object
        The atlas objects contains the maps in the `maps` key and the labels in the `labels` key.

    See Also
    --------
    graphpype.graph.randomSpin : Spin distributions use atlas data to remove spatial autocorrelation from fmri signals.

    """
    # check the atlas exists
    if os.path.exists(atlasDir + atlas):
        atlasObj = loadObj(atlasDir + atlas)
    else:
        # fetch the atlas and labels
        funcStr = "datasets.fetch_atlas_" + atlas
        funcFetch = getattr(nilearn, funcStr)
        atlasObj = funcFetch()
        # seperate the regions into spatially continuous blocks (no left-right hemisphere symmetry for example)
        atlasObj = connected_label_regions(atlasObj)
        # atlasMaps = atlasObj["maps"]
        # atlasLabels = atlas["labels"]
        
        saveObj(atlasObj, atlasDir)

    return atlasObj

def fmriprep(directory, fmriprep=[], participant=[], cache=True):
    r"""An API call to the fmriprep preprocessing package.

    All fMRI and structural data should be preprocessed before it can be converted into a graph datum object. The fmriprep API is a popular, but not unique method, of doing this. See also: freesurfer and nipype APIs. The fmriprep API call is constructed with a number of flags and a processed over a number of participant paths.

    Parameters
    ---------- 
    directory : string
        The root BIDS data directory. All processed data will be placed in the 'derivatives' subfolder to remain BIDS compliant.
    fmriprep : list
        A list of strings for the flags used in the fmriprep pipeline.
    participant: list
        A list of indexes/strings to flag the subjects to process. An index will be interpreted as the unix indexed subject while a string will match the subject path e.g. 0400. An empty list defaults to selecting all available participants.
    cache: bool
        If true the preprocessed result is cached allowing the pipeline to be interupted.

    Returns
    -------
    None

    """
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
    r"""A simple loading function for pickled objects.
        
    Parameters
    ----------
    directory : string
        The path of the saved object.

    Returns
    -------
    obj : object

    """
    with open(directory, 'wb') as file:
        obj = pickle.load(file)
    return obj

def saveObject(obj, directory):
    r"""A simple saving utility function for python objects using pickle.

    Parameters
    ----------
    obj : object
        Object to be pickled/saved.
    directory : string
        Path for pickled object to be saved.
    
    Notes
    -----
    For BIDS compliance it is recommended that the data be saved in the derivatives subfolder 'derivatives/graphpype/objects'.

    Raises
    ------
    Warning
        Alert the user to non-BIDS compliance.
    """

    if 'derivatives' not in directory:
        warnings.warn("The save path does not appear to be in the derivatives subdirectory. Remember to make sure that your analysis is BIDS compliant.")

    with open(directory, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def loadParcellation(*totalDataset, dataDirectory="", listFilterDirectory="", channel="", nameFilter=False):
    r"""Load a pre-parcellated dataset which may be filtered into multiple datasets.

    Parameters
    ----------
    totalDataset : list
        The analysis dataset of the pipeline.
    dataDirectory : string
        The location of the processed data.
    listFilterDirectory : string
        The location of the dataset names to filter by. If empty or "" then sorting occurs numerically on the first index of data.
    channel : string
        The channel key which the processed data will be assigned to e.g. "corticalThick"
    nameFilter : bool
        When true the processed data will be sorted into the datasets in the pipeline with matching names. Otherwise, sorting occurs by numerical index.
    
    Returns
    -------
    None
        Dataset objects are modified inplace.
    
    Notes
    -----
    
    Data is often processed via a particular parcellation and recorded as part of a named subgroup and combined into a single file. This function allows these files to be processed into multiple datasets with graph features determined by the processed parcellation e.g. cortical thickness at a particular node. It is most useful when not working directly with imaging data. It is an analysis level data loader that acts in the preprocessing stage of the pipeline bypassing the datum processing level. Data is expected to come in the form of a matrix of floats at `dataDirectory`, a (possibly empty) list of strings at `listFilterDirectory`, and a pipeline dataset.
    """
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
    r"""Loads saved data into a data channel at the analysis level.

    Currently supports numpy tensor loading via CSV and npy file types.

    Parameters
    ----------
    totalDataset : graphpype.pipe.DataSet
        A vector containing the datasets which the dataset will be loaded to. Typically length one, at the post-analysis level.
    dataDirectory : string
        Path to the data.
    channel : string
        Channel which the data should be asscociated with in the analysis.
    dataType : string
        How the data is saved.
    
    Returns
    -------
    None
        Data is modified inplace.

    Notes
    -----
    The loading of numpy tensors is supported through 'npy' and 'csv' strings in the ``dataType`` variable. If no datatype is specified python will attempt to open the file but no further support will be given on how to read the file and this should be processed in a function downstream.
    
    """
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
    r"""Construct a distance matrix from a list of coordinates.

    The data can be two or three dimensional.

    Parameters
    ----------
    data : list
        A list of geometric coordinates.

    Returns
    -------
    mat : numpy.ndarray
        A square matrix in the form of a numpy array. 
    
    Notes
    -----
    The distance metric used is the euclidean metric.
    """
    L = len(data)
    mat = numpy.zeros((L,L))
    for i in range(L):
        for j in range(L):
            mat[i,j] = numpy.sqrt(numpy.sum((data[i] - data[j]) ** 2))
    return mat

def vectorSlice(tensor, index, axis):
    r"""Take an indexed axis slice out of a tensor and reshape into a single vector.
    
    Parameters
    ----------
    tensor : numpy.ndarray
    index : tuple
        The indexes through which the slices of the tensor should be taken.
    axis : int
        The dimension along which to slice.

    Returns
    -------
    numpy.ndarray

    """
    assert type(tensor) == numpy.numpy.ndarray
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
