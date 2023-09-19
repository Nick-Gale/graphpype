# Submodule: pipelines. Functions dedicated to constructing pipelines: pre-processing, processing, plotting, and end-to-end functionality.
import subprocess as sp
# Constants

# Classes

class Recipe:
    """Pipeline Recipe: a directed line graph of functions/constants that operate on a subset of data for analysis, output, and plotting. The recipe informs the operations that must be sequentially applied to data and subsequent output manipulations. Recipes can be composed by concatentation."""

    name: str
    """Recipe identifier"""

    description: str
    """Summary of the pipeline flow i.e. brief description of what the recipe cooks."""

    nodes: list
    """List of functions that the analysis will operate with"""

    env: dict
    """Any enviroment variables including number of threads, randomisation seed, etc."""

    def __init__(self, name: str, description: str, nodes: list["Operator"] = [], seed: int=1):
        """Initialise a (potentially empty) recipe card with a name and descriptor."""

        # TO DO

    def check(self, DataType):
        """Check that for a given dataset the recipe is well defined."""

        # Broadcast type over list of data and check all data types are the same (expected: list of tensors of Floats or Ints).

        # For each node in recipe check inputs match outputs of previous node and check first input matches the data type.

        # Report the first break in the chain else report fine.

    def report(self, data):
        """Generate a report card summarising the recipe"""

class Datum:
    """An object that completely specifies a particular element of the dataset with potentially multiple imaging modalities."""

    dirs: list
    """List of the directories used for the specimen in the analysis."""
    preData: dict
    """Dictionary storing the preprocessed data from a particular pipeline such as prepfmri e.g. Dict["Connectivity"] = matrix"""
    postData: dict
    """Dictionary storing the processed data from a particular operator e.g. Dict["Connectivity"] = matrix"""
    
    def __init__(self, directories*)
        dirs = []
        preData = dict()
        # list the directories
        for d in directories:
            append.dirs(d["string"])
        
        # split the directories to find the relevant information
        for d in directories:
            channel = d["channel"]

            if channel == "Gene":
                data = np.load(d["string"])
            elif channel = "fMRI":
                data = nib.load(d["string"]).get_fdata()
            else:
                raise(AssertionError, "The specified data type is either not supported or not in your BIDS directory.")
            
            preData[d["channel"]] = data

    def process(op=Operator):
        channelData = [preData[i] for i in op["channelsIn"]]
        processedData = op(channelData*, op["arguments"])
        postData[op["channelOut"]] = processedDatazo

    
class DataSet:
    """A composition of data with type `Datum`. Contains dataset level analysis and processing."""

    data: list
    """The processed data."""
    analysis: dict
    """Group level analysis of processed data."""
    
    def __init__(self, dataobjs*):
        data = []
        for i in dataobjs:
            append.data(i)

    def process(op=Operator):
        channeledData = [[i.c for i in data] for c in op["channelsIn"]]
        processedData = op(channeledData*, op["arguments"])
        analysis[op["channelOut"]] = processedData
    
class Operator:
    """A processing operator that operates on a discrete chunk of data which either is broadcastover/reduces a data set. The result is stored in the DataType object in a channel indexed as a dictionary and specified by this operator. 

    Usage: Operator(["name": "randn", "package": "numpy.random"], channels)

    """
    name: str
    basePackage: str
    """Give the package name."""
    packageDir: str
    version: str
    """Give the version number for the locally installed package. If the function is self defined then provide the relative directory."""
    arguments: dict
    """The arguments required to run the operator. Unnamed arguments should be assigned to the dictionary entry unnamed."""
    channelsIn: list
    """The shape of the container of the incoming data."""
    channelsOut: list
    """The shape of the container of the returned data."""
    def __init__(self, function=dict, channels=dict, args=dict, data=None, local=False):

        assert channels["dataIndex"] >= 0, "You must supply a data index in the form of an integer >0 for each operator."""
        
        assert channels["resultIndex"] >= 0, "You must supply an index which will be asscociated with the results."""
        
        assert type(function["name"])==str, "You must supply a function name."
        name = function["name"]
        
        if local:
            version = local
        else:
            if "." in function["package"]:
                basePackage = function["package"][0:function["package"].index(".")]
                packageDir = function["package"][function["package"].index(","):-1]
            else:
                basePackage = function["package"]

            if function["version"]
                version = function["version"]
                # check that locally installed version exists

                # check that function exists

            # print version string of python package

        arguments = args

        if data:
            assert type(data) == type(np.array()), "The input data is not in the form of a numpy array."
            shapeIn = data.shape
            
    def __call__(self, data):


# Functions

# # # Piping
def _preprocess(recipe, bidsdir):
    """Apply a preprocessing pipeline to a directory of data in the BIDS standard. Currently only command line based preprocessing pipes are supported. A user defined pipeline can be specified using a series of operators on a list of directories."""
    
    ops = recipes.nodes["preProcess"]
    for i in ops:
        if i[packageDir]["cmd"]:
            cmdStr =[k + v for (k, v) in i["arguments"]].prepend(i["name"])
            sp.run(cmdStr)

def _process(recipe, bidsdir):
    """Process all subject data with no intersubject dependencies."""

    ops = recipes.nodes["postProcess"]
    if recipe.env["nThreads"] > 1:
        # parallelise
    else:
        for i in ops:
            

def _pipe(recipe, data, cache_boole, cache_directory)
    if cache_boole:
        return _cache_data_pipe(recipe, data, cache_directory)
    else:
        return _data_pipe(recipe, data)

def _data_pipe(recipe, data):
    result = data
    for f in recipe.nodes:
        data = f(data)
    return data

