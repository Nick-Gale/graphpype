# Submodule: pipelines. Functions dedicated to constructing pipelines: pre-processing, processing, plotting, and end-to-end functionality.
import importlib, warnings, os
import subprocess as sp
# Constants

# Classes

class Pipeline:
    """Piping object."""
    recipe: any
    paths: list
    result: any
    def __init__(self, recipeDir, *bids):
        recipe = Recipe()
        recipe.fromDir(recipeDir)
        self.recipe = recipe
        
        # for each dataset directory find the subject folder names
        subjPaths = []
        for i in bids:
            subjPaths.append( [ f.path for f in os.scandir(folder) if f.is_dir() and f.path[0:4] == 'sub-' ] )

        self.paths = subjPaths

    def process():
        datasets = []

            
        # preprocess: check if already preprocessed
        #               preprocess
        #               load into datasets, one per directory, or apply some other filtering
        for i in bids:
            _preprocess(recipe, i)
        
        # construct dataset
        for s in s:
            ds = [Datum(j) for j in s]
            datasets.append(DataSet(ds))
        
        analysisSet = DataSet(datasets)

        # process: check cache
        #           process nodes
        #           cache
        if recipe.nodes["postProcess"]:
            ops = recipe.nodes["postProcess"]
            for d in datasets:
                _process(ops, d.data, self.recipe.env["nThreads"])

        if recipe.nodes["analysis"]:
            ops = recipe.nodes["analysis"]
            _process(ops, datasets, self.recipe.env["nThreads"])
        
        if recipe.nodes["postAnalysis"]:
            ops = recipe.nodes["postAnalysis"]
            _process(ops, analysisSet, self.recipe.env["nThreads"])
    
    def output(outputDir):
        # output is classed differently to processing because it is likely to change often and the whole processing pipe needn't be run multiple times
        assert outputDir[0] != '/', "Please use relative directories, not absolute."
        
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)
        
        # change to specified output directory
        os.chdir(os.getcwd() + outputDir)
        ops = self.recipe.nodes["output"]
        for c in ops:
            _process(ops, self.result, self.recipe.env["nThreads"]) 

        if outputDir:
            os.chdir(os.getcwd() + '../')
class Recipe:
    """Pipeline Recipe: a directed line graph of functions/constants that operate on a subset of data for analysis, output, and plotting. The recipe informs the operations that must be sequentially applied to data and subsequent output manipulations. Recipes can be composed by concatentation."""

    name: str
    """Recipe identifier"""

    description: str
    """Summary of the pipeline flow i.e. brief description of what the recipe cooks."""

    nodes: dict 
    """Functions that the analysis will operate with. The entries are: "preProcess", "postProcess", "analysis", "postAnalysis", and "output". Analysis refers to functions that are applied to a specific data set, while postAnalysis refers to functions that are applied to multiple datasets. Plotting, or output functions such as reports and caching go in the "output" layer."""

    env: dict
    """Any enviroment variables including number of threads, randomisation seed, etc."""

    def __init__(self, name: str="", description: str="", nodes: dict={}, env: dict={"seed": 1, "nThreads": 1}):
        """Initialise a (potentially empty) recipe card with a name and descriptor."""
        self.name = name
        self.description = description
        self.nodes = nodes
        self.env = env

    def report(self, data):
        """Generate a report card summarising the recipe"""

            ### DO THIS ###
    def write(self, outputDir):
        """Write the recipe to disk in .json format for use in pipelines"""
        import json
        with open(outputDir, 'w') as f:
            json_obj = {"name": self.name, "description": self.description, "nodes": [i.json for i in self.nodes], "env": self.env}
            json.dump(json_obj, f, ensure_ascii=False, indent=4)
    
    def read(self, inputDir):
        """Read a recipe in .json format from disk and construct recipe object."""
        import json
        with open(inputDir, 'r') as f:
            recipe = json.load(f)
        self.name = recipe["name"]
        self.description = recipe["description"]
        print(recipe["nodes"])
        self.nodes = [Operator(**i) for i in recipe["nodes"]]
        self.env = recipe["env"]

class Datum:
    """An object that completely specifies a particular element of the dataset with potentially multiple imaging modalities."""

    dirs: list
    """List of the directories used for the specimen/subject in the analysis."""
    preData: dict
    """Dictionary storing the preprocessed data from a particular pipeline such as prepfmri e.g. Dict["Connectivity"] = matrix"""
    postData: dict
    """Dictionary storing the processed data from a particular operator e.g. Dict["Connectivity"] = matrix"""
    
    def __init__(self, directories):
        self.dirs = []
        self.preData = {} 
        self.postData = {} 
        # list the directories
        for d in directories:
            self.dirs.append(d["string"])
        
        # split the directories to find the relevant information
        for d in directories:
            channel = d["channel"]

            if channel == "Gene":
                import numpy as np
                data = np.load(d["string"])
            elif channel == "fMRI":
                import nibabel as nib
                data = nib.load(d["string"]).get_fdata()
            elif channel == "Random":
                from numpy.random import rand
                data = rand(5,2,4,10)
            else:
                raise(AssertionError, "The specified data type is either not supported or not in your BIDS directory.")
            
            self.preData[d["channel"]] = data
    def __call__(self, directory):
        # append to channel directory

class DataSet:
    """A composition of data with type `Datum`. Contains dataset level analysis and processing."""
    name: str
    """A name for the dataset (default: "")"""
    data: list
    """The processed data."""
    analysis: dict
    """Group level analysis of processed data."""
    
    def __init__(self, name, dataobjs):
        name=name
        data = []
        analysis = {}
        for i in dataobjs:
            append.data(i)
    def __call__(self, directory, channel, loader):
        # append to the analysis
        
class Operator:
    """A processing operator that operates on a discrete chunk of data which either is broadcastover/reduces a data set. The result is stored in the DataType object in a channel indexed as a dictionary and specified by this operator. 
    

    Usage: Operator(function, channels, arguments, inter)
        To specify the function:
        function = {"name": name of the function,
                    "package": e.g numpy.random,
                    "version": blank if installed package / directory of user defined function}
        
        channels ={{"dataIndex": {"Layer": ["Channel1", "Channel2", etc]}, {"Layer2": ["Channel0"]}}, 
                    {"resultIndex": {"Layer": ["SingleChannel"]}}}
        
        arguments = {"unnamed" = [], "alpha": 0, "beta": 1}

        inter = {}, {"broadcast": True}
        The default is to apply the operator to the function channel as is but occasionally you might want to broadcast over a list of elements in the channel e.g. doing a spin correction.

    """
    name: str
    basePackage: str
    """Give the package name."""
    packageDir: str
    version: str
    """Give the version number for the locally installed package. If the function is self defined then provide the relative directory."""
    arguments: dict
    """The arguments required to run the operator. Unnamed arguments should be assigned to the dictionary entry unnamed."""
    internal: dict
    """A dictionary of internal operating requirements e.g. broadcast, reduce. Broadcast and reduce will always be applied to the first index of the data channels"""
    channelsIn: list
    """The shape of the container of the incoming data. The first entry specifies the layer on which the operator should function"""
    channelOut: list
    """The shape of the container of the returned data."""
    json: list
    def __init__(self, function=dict, channels=dict, args=dict, inter={}):

        assert channels["dataIndex"], "You must supply a data channel/s in the form of a dictionary keys for each operator. The operator will broadcast over the supplied channels."
        self.channelsIn = channels["dataIndex"]
        
        assert channels["resultIndex"], "You must supply a layer and data channel  which will be asscociated with the results."
        self.channelOut = channels["resultIndex"]

        assert type(function["name"])==str, "You must supply a function name."
        self.name = function["name"]
        
        if ("local" in function) and (function["local"] == 1):
            self.version = function["local"]
        elif function["local"] == 0:
            self.basePackage = ""
            self.packageDir = ""
            if "." in function["package"]:
                self.basePackage += function["package"][0:function["package"].index(".")]
                self.packageDir += function["package"][function["package"].index("."):-1]
            else:
                self.basePackage = function["package"]

            if function["version"]:
                self.version = function["version"]
            else:
                pkg = importlib.import_module(self.basePackage)
                if pkg:
                    if pkg.__version__:
                        self.version = pkg.__version__
                    elif importlib.metadata(basePkg):
                        self.version = importlib.metadata(basePkg)
                else:
                    warnings.warn("Couldn't find a version for the function being called: setting version to 0.0.0", UserWarning)
                    self.version = "0.0.0"

        self.arguments = args
        
        self.internal = inter
        self.json = {"function": function, "channels": channels, "args": args, "inter": inter} 
    def __call__(self, data=Datum):
        """The operator can be used to operate on a datum by specifying the data layers and channels. The data will be stored in a strictly ordered vector which will be passed as a tuple to the function which defines the operator with the order being inherited from the order used to specify the channels. The result is stored in a single channel in a layer specified by the `channelOut` field."""

        if self.packageDir == "":
            pkg = importlib.import_module(self.basePackage)
        else:
            pkg = importlib.import_module(self.basePackage + "." + self.packageDir)
        f = getattr(pkg, self.name)
        
        d = []
        for i in self.channelsIn:
            if i == "preData":
                for g in self.channelsIn[i]:
                    d.append(data.preData[g])
            elif i == "postData":
                for g in self.channelsIn[i]:
                    d.append(data.postData[g])
            else:
                raise NameError("There is no valid layer by that name.")
        
        if self.internal["broadcast"]:
            # always broadcast over the first provided channel
            for i in range(d[0]):
                v = [d[x][i % len(d[x])] for x in d]
                res = f(*v, **self.arguments)
        else:
            if self.arguments["unnamed"]:
                v = self.arguments["unnamed"]
                res = f(*d, *v, **self.arguments)
            else:    
                res = f(*d, **self.arguments)
        
        for c in self.channelOut:
            if c == "preData":
                data.preData[self.channelOut[c]] = res
            if c == "postData":
                data.postData[self.channelOut[c]] = res

        
    def __call__(self, data=DataSet):
        if self.packageDir == "":
            pkg = importlib.import_module(self.basePackage)
        else:
            pkg = importlib.import_module(self.basePackage + "." + self.packageDir)
        f = getattr(pkg, self.name)
        
    
        for i in self.channelsIn:
            if i == "preData":
                for g in self.channelsIn[i]:
                    d = [x.preData[g] for x in data.data]
  
            elif i == "postData":
                for g in self.channelsIn[i]:
                    d = [x.postData[g] for x in data.data]
            
            elif i == "analysis":
                ### TO DO: ensure this works for a single data set as well as a list of datasets (recommended)
                for g in self.channelsIn[i]:
                    d = [x.analysis[g] for x in data]
            else:
                raise NameError("There is no valid layer by that name.")

        if self.internal["broadcast"]:
            # always broadcast over the first provided channel
            for i in range(d[0]):
                v = [d[x][i % len(d[x])] for x in d]
                res = f(*v, **self.arguments)
        else:
            if self.arguments["unnamed"]:
                v = self.arguments["unnamed"]
                res = f(*d, *v, **self.arguments)
            else:
                res = f(*d, **self.arguments)

        data.analysis[self.channelOut["analysis"]] = res
# Functions

# # # Piping
def _preprocess(recipe, bidsdir):
    """Apply a preprocessing pipeline to a directory of data in the BIDS standard. Currently only command line based preprocessing pipes are supported. A user defined pipeline can be specified using a series of operators on a list of directories."""
    
    ops = recipes.nodes["preProcess"]
    for i in ops:
        if i[packageDir]["cmd"]:
            cmdStr =[k + v for (k, v) in i["arguments"]].prepend(i["name"])
            sp.run(cmdStr)

def _process(ops, dataset, parallel: int):
    if parallel > 1:
        import multiprocessing as mp
        pool_obj = mp.Pool(nThreads)
        for i in ops:
            pool_obj.map(i, dataset) # check that this is memory safe
    else:
        for d in dataset:
            [f(d) for f in ops]


            

