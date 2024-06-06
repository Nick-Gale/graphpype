# Submodule: pipelines. Functions dedicated to constructing pipelines: pre-processing, processing, plotting, and end-to-end functionality.
"""
Pipelines
=========
A collection of functions and object classes to construct recipes and run analysis pipelines.

The major conceptual classes exported are:
    1. Datum - graphpypes internal expression of data containing metadata fields, preprocessing, and post-processing analysis of raw data.
    2. Dataset - graphpypes internal represtation of datasets comprising of a list of data objects and analysis. Is composable with itself.
    3. Operator - graphpypes internal representation of functions that operator on data and datasets. Contains metadata to aid with reproducibility and portability.
    4. Recipe - an object to specify how operators should interact with data.
    5. Pipeline - an object to construct and analyse data through a structured pipeline defined by the recipe.
"""


import importlib, warnings, os, time
import subprocess 
import graphpype
# Constants

# Classes

class Pipeline:
    r"""

    Pipeline object: takes a recipe and executes it over a series of datapaths.
    
    Attributes
    ----------
    recipe : graphpype.pipe.recipe
        The recipe for the pipeline to execute.
    paths : list
        The BIDS paths for data.
    result : graphpype.pipe.dataset
        A dataset containing the post-analysis, individual dataset analyses, post-processing of each datum, and preprocessing routines.


    """
    recipe: any
    paths: list
    result: any
    def __init__(self, recipeDir, bids = {}):
        r"""Initialise a pipeline object."""
        recipe = Recipe()
        recipe.read(recipeDir)
        self.recipe = recipe
        
        # for each dataset directory find the subject folder names
        subjPaths = {}
        if bids:
            for (dataSetName, directory) in bids.items():
                subjPaths[dataSetName] = [ f.path for f in os.scandir(directory) if f.is_dir() and 'sub-' in f.path]
                subjPaths[dataSetName].insert(0, directory)
        self.paths = subjPaths

    def process(pipe, dataSetNames=[], preProcess=True):
        r"""Process a pipeline over a series of datasets.

        Parameters
        ----------
        dataSetNames : list
            A list of strings defining the datasets to process.
        preProcess : bool, optional
            Optional, but recommended, preprocessing.
        """
        processingStart = time.time()
        datasets = []
        if preProcess:
            for (dataSetName, paths) in pipe.paths.items():
                _preprocess(pipe.recipe, paths[0])
            
            # construct dataset THIS NEEDS TO BE FIXED
            for (dataSetName, paths) in pipe.paths.items():
                # since the data is preprocessed it needs to be in the derivatives folder and paths must be accordingly modified
                
                directory = paths[0] + "/derivatives/"
                os.mkdir(directory) if (not os.path.isdir(directory)) else None
                dataPaths = [ f.path for f in os.scandir(directory) if f.is_dir() and 'sub-' in f.path ]
                ds=[]
                for p in dataPaths:
                    dirsP = []
                    for o in pipe.recipe.nodes["preProcess"]:
                        channel = o.channelOut["preProcess"][0]
                        if "stringAdd" in o.args:
                            string = p + o.args["stringAdd"]
                        else:
                            string = p
                        dirsP.append({"channel": channel, "string": string})
                    ds.append(Datum(*dirsP))

                datasets.append(DataSet(name=dataSetName, dataObjs=ds))
        
        else:
            if dataSetNames:
                [datasets.append(DataSet(name=i, dataObjs=[])) for i in dataSetNames]
            else:
                ### CHECK THIS
                [datasets.append(DataSet(i, [])) for i in pipe.recipe.nodes if "load" in i.channelsIn]
        
        analysisSet = DataSet(dataObjs=datasets)
        # process: check cache
        #           process nodes
        #           cache
        

        if "preProcess" in pipe.recipe.nodes:
            # do the preprocessing that needs to be applied to the whole data set (e.g. loading already preprocessed data)
            ops = [o for o in pipe.recipe.nodes["preProcess"] if "totalAnalysisPreProcessing" in o.internal]
            res =  _process(ops, [analysisSet], pipe.recipe.env["nThreads"])
            if res:
                analysisSet = res[0]
             
        if "postProcess" in pipe.recipe.nodes:
            ops = pipe.recipe.nodes["postProcess"]
            res = _process(ops, analysisSet.data, pipe.recipe.env["nThreads"])
            if res:
                analysisSet.data = res
        
        if "analysis" in pipe.recipe.nodes:
            ops = pipe.recipe.nodes["analysis"]
            res = _process(ops, analysisSet.data, pipe.recipe.env["nThreads"])
            if res:
                analysisSet.data = res 
        
        if "postAnalysis" in pipe.recipe.nodes:
            ops = pipe.recipe.nodes["postAnalysis"]
            res = _process(ops, [analysisSet], pipe.recipe.env["nThreads"])
            if res:
                analysisSet = res[0]

        pipe.result = analysisSet
        processingFinish = time.time()
        print(f"Total processing time: {processingFinish - processingStart} seconds")

    def output(outputDir):
        """Generate the output of the analysis."""
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

    def plot(self, outputDir):
        """Generate plot files for the plotting objects defined in the analysis.""" 
        import matplotlib.pyplot as plt
        
        plotObjs = self.result.analysis["plots"][0]
        fileNames = self.result.analysis["plots"][1]

        for i in range(len(plotObjs)):
            plotObjs[i].savefig(outputDir + fileNames[i])


class Recipe:
    r"""
    Pipeline Recipe: a directed line graph of functions/constants that operate on a subset of data for analysis, output, and plotting. 

    The recipe informs the operations that must be sequentially applied to data and subsequent output manipulations. Recipes can be composed by concatentation.

    Attributes
    ----------
    name: str
        Recipe identifier
    description: str
        Summary of the pipeline flow i.e. brief description of what the recipe cooks.
    nodes: dict 
        Functions that the analysis will operate with. The entries are: "preProcess", "postProcess", "analysis", "postAnalysis", and "output". Analysis refers to functions that are applied to a specific data set, while postAnalysis refers to functions that are applied to multiple datasets. Plotting, or output functions such as reports and caching go in the "output" layer.
    env: dict
        Any enviroment variables including number of threads, randomisation seed, etc.

    """
    name: str
    description: str
    nodes: dict 
    env: dict
    def __init__(self, name: str="", description: str="", nodes: dict={}, env: dict={"seed": 1, "nThreads": 1}, template=""):
        """Initialise a (potentially empty) recipe card with a name and descriptor.

        Some templates are provided: empty, nipype, and full.

        Parameters
        ----------
        name : str
        description : str
        nodes : dict
            The nodes of the recipe expressed as graphpype.Operator objects in dictionary labels: preProcess, postProcess, analysis, postAnalysis
        env : dict
            Specifies any environment variables such as seed, or number of threads.
        template : str
            Generates a template recipe from the 'templates' directory.
        """

        self.name = name
        self.description = description
        self.nodes = nodes
        self.env = env
        
        if template != "":
            assert template in ["empty", "full", "nipype"], "Please provide a valid template."
            # get the templates path
            import sys
            srcPath = sys.modules["graphpype"].__file__[0:-11]
            templatePath = srcPath + "../../templates/" + template + ".json"
            
            import json
            with open(templatePath, 'r') as f:
                recipe = json.load(f) 
            
            self.name = recipe["name"]
            
            self.description = recipe["description"]
            
            self.nodes = {}
            for n in recipe["nodes"]:
                ops = [Operator(**i) for i in recipe["nodes"][n]]
                self.nodes[n] = ops
            
            self.env = recipe["env"]

    def report(self, bids, author="", outputDir="data/derivatives"):
        """Generate a report card summarising the recipe.
        

        """
        assert outputDir[0] != '/', "Please use relative directories, not absolute."         
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)
        
        titleStr = "# graphpype Recipe : " + self.name + "\n" # + " <br> "

        bodyStr = self.description + "\n"

        import datetime as dt
        dateStr = (dt.datetime.now(dt.UTC)).strftime('%Y-%m-%d')
        if author == "":
            authorStr = "not specified"
            warnings.warn("You haven't specified the author.")
        endStr = "\n<i>This report was generated on " + dateStr + ". The author was " + authorStr + ".</i> \n"

        block0 = titleStr + bodyStr + endStr + "\n"
        
        # Data
        dataStr = "## Data \n"
        numberType = ""
        for (dataSetName, directory) in bids.items():
            subjPaths = [ f.path for f in os.scandir(directory) if f.is_dir() and 'sub-' in f.path ]
            numberType += "The " + dataSetName + " dataset is located in the directory " + directory + f". There are {len(subjPaths)} files in this directory."+ " <br> "
        block1 = dataStr + numberType + " \n "
        
        # Operators
        lenO = []
        operatorStr = "## Operators \n"
        tableStr = ""
        for level, operators in self.nodes.items():
            lenO.append(len(operators))
            operatorTitleStr = "\n### Operator Level: " + level + " \n"
            operatorTableStr = "<table border=\"1px solid black\">\n<thead>\n<trow class=\"firstLine\">\n <th>Name</th>\n <th>Description</th>\n <th>Input: Data Channels (Level)</th>\n <th>Output: Data Channels (Level)</th>\n</tr>\n</thead>\n<tbody>"
            # operatorDivider =  "| ---- | ----------- | ---------------------------- | ----------------------------- | \n <br>"
            nodeStr = ""
            for n in operators:
                nodeStr += "\n<tr>\n <td><i>" + n.name + "</i></td>\n <td>" + n.description + "</td>\n <td>"
                for (key, val) in n.channelsIn.items():
                    if val: 
                        for v in val:
                            nodeStr += "<i>" + v + "</i>" + " (" + key + "), "
                    else:
                        nodeStr += "N/A "
                
                nodeStr += "</td>\n <td>"

                for (key, val) in n.channelOut.items():
                    for v in val:
                        nodeStr += "<i>" + v + "</i>" + " (" + key + "), "
                nodeStr += "</td>\n</tr>"
        #        nodeStr += operatorDivider
            # tableStr += operatorTitleStr + operatorTableStr + operatorDivider + nodeStr + "\n" 
            tableStr += operatorTitleStr + operatorTableStr + nodeStr + "</tbody>\n</table>\n" 
        
        nNodesStr = f"There are {lenO[0]} pre-processing operators, {lenO[1]} post-processing operators {lenO[2]} analysis operators, and {lenO[3]} post-Analysis operators.\n"
        
        block2 = operatorStr + nNodesStr + tableStr 
        #Flowchart
        flowchart = graphpype.utils.generateFlowchart(self)
        flowchart.savefig(outputDir + "/flowchart.png")
        chartTitle = "## Operational Flow Chart \n"
        imgStr = "<img src=\"./flowchart.png\"><figcaption>Flowchart of the analysis pipeline generated by graphpype.</figcaption>"
        block3 = chartTitle + imgStr 
        # Convert markdown to HTML and save
        markdownStr = block0 + " \n" + block1 + " \n" +  block2 +  " \n" + block3 + " \n"
        
        import markdown
        htmlObj = markdown.markdown(markdownStr)
        htmlDir = outputDir + "/reportCard.html" 
        with open(htmlDir, 'w') as f:
            f.write(htmlObj)
        print("Check the output directory: " + outputDir)

    def write(self, outputDir):
        """
        Write the recipe to disk in .json format for use in pipelines.
        
        Parameters
        ----------
        outputDir : str
            The path to write the recipe to.

        Returns
        -------
        None
        """
        import json
        with open(outputDir, 'w') as f:
            # print())
            # json_obj = {"name": self.name, "description": self.description, "nodes": sum([[j.json for j in self.nodes[i]] for i in self.nodes], []), "env": self.env}
            nNodes = {}
            for i in self.nodes:
                nNodes[i] = [j.json for j in self.nodes[i]]
            
            json_obj = {"name": self.name, "description": self.description, "nodes": nNodes, "env": self.env}
            json.dump(json_obj, f, ensure_ascii=False, indent=4)
    
    def read(self, inputDir):
        r"""
        Read a recipe in .json format from disk and construct recipe object.

        Parameters
        ----------
        inputDir : str
            The path to a recipe.
        """
        import json
        with open(inputDir, 'r') as f:
            recipe = json.load(f) 
        
        self.name = recipe["name"]
        
        self.description = recipe["description"]
        
        self.nodes = {}
        for n in recipe["nodes"]:
            ops = [Operator(**i) for i in recipe["nodes"][n]]
            self.nodes[n] = ops
        
        self.env = recipe["env"]

class Datum:
    r"""
    An object that completely specifies a particular element of the dataset with potentially multiple imaging modalities.

    Attributes
    ----------

    dirs: list
    List of the directories used for the specimen/subject in the analysis.
    preData: dict
    Dictionary storing the preprocessed data from a particular pipeline such as prepfmri e.g. Dict["Connectivity"] = matrix
    postData: dict
    Dictionary storing the processed data from a particular operator e.g. Dict["Connectivity"] = matrix

    """ 

    dirs: list
    preData: dict
    postData: dict
    
    def __init__(self, *directories):
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
            elif channel == "fMRI" or channel == "fmri" or channel == "FMRI":
                allNiFTi = [ f.path for f in os.scandir(d["string"]) if f.path[-6:] == 'nii.gz' ]
                import nibabel 
                if d["string"][-1] == '/':
                    print("No specific file selected in data subdirectory, choosing the NIFTI .gz found at the 0'th index of a string matched to `desc-preproc_bold.nii.gz`")
                    target = 'desc-preproc_bold.nii.gz'
                    preProcessedString = [s for s in allNiFTi if s[-len(target):] == target][0]
                    data = nibabel.load(preProcessedString).get_fdata()
                else:
                    assert d["string"][-6:] == 'nii.gz', "Please enter a valid NIFTI archive file"
                    data = nibabel.load(d["string"] + '/nifti.gz').get_fdata()
                
                
            elif channel == "Random":
                from numpy.random import rand
                data = rand(5,2,4,10)
            elif channel == "none":
                data = []
            else:
                raise(AssertionError, "The specified data type is either not supported or not in your BIDS directory.")
            
            self.preData[d["channel"]] = data
    def __call__(self, directory):
        return None
        # append to channel directory
    def addChannel(self, channel, data):
        self.preData[channel] = data
        
class DataSet:
    r"""
    A composition of data with type `Datum`. Contains dataset level analysis and processing.
    
    Attributes
    ----------
    name: str
        A name for the dataset (default: "")
    data: list
        The processed data.
    analysis: dict
        Group level analysis of processed data.

    """
    name: str
    data: list
    analysis: dict
    
    def __init__(self, name="", dataObjs=[]):
        self.name = name
        self.data = []
        self.analysis = {}
        for i in dataObjs:
            self.data.append(i)
    def __call__(self, directory, channel, loader):
        return None
        # append to the analysis
        
class Operator:
    """A processing operator that operates on a discrete chunk of data which either is broadcastover/reduces a data set. The result is stored in the DataType object in a channel indexed as a dictionary and specified by this operator. 
    
    Attributes
    ----------
    opName: str
        A local understandable name.
    description: str
        A description of what the operator does.
    name: str
        Name of the function.
    basePackage: str
        The package name.
    packageDir: str
    version: str
        Give the version number for the locally installed package. If the function is self defined then provide the relative directory.
    args: dict
        The args required to run the operator. Unnamed arguments should be assigned to the dictionary entry unnamed.
    internal: dict
        A dictionary of internal operating requirements e.g. broadcast, reduce. Broadcast and reduce will always be applied to the first index of the data channels.
    channelsIn: list
        The shape of the container of the incoming data. The first entry specifies the layer on which the operator should function.
    channelOut: list
        The shape of the container of the returned data.
    json: list
        A JSON object storing the object.

    Notes
    -----
    The default is to apply the operator to the function channel as is but occasionally you might want to broadcast over a list of elements in the channel e.g. doing a spin correction. To specify the function:
    - function = {"name": name of the function, "package": e.g numpy.random, "version": blank if installed package / directory of user defined function}
    - channels ={{"dataIndex": {"Layer": ["Channel1", "Channel2", etc]}, {"Layer2": ["Channel0"]}}, {"resultIndex": {"Layer": ["SingleChannel"]}}}
    - args = {"unnamed" = [], "alpha": 0, "beta": 1}
    - inter = {}, {"broadcast": True}
    """
 #     opName: str
 #     description: str
 #     name: str
 #     basePackage: str
 #     packageDir: str
 #     version: str
 #     args: dict
 #     internal: dict
 #     channelsIn: list
 #     channelOut: list
 #     json: list
    def __init__(self, name: str, description: str, function: dict, channels: dict, args: dict, inter={}):

        assert channels["dataIndex"], "You must supply a data channel/s in the form of a dictionary keys for each operator. The operator will broadcast over the supplied channels."
        self.channelsIn = channels["dataIndex"]
        
        assert channels["resultIndex"], "You must supply a layer and data channel  which will be asscociated with the results."
        self.channelOut = channels["resultIndex"]

        assert type(function["name"])==str, "You must supply a function name."
        self.name = function["name"]

        self.opName=name
        self.description=description
        
        if ("local" in function) and (function["local"] == 1):
            self.version = function["local"]
            self.packageDir = "local" # THINK ABOUT THIS, PERHAPS USERS WANT TO SPECIFY WITHIN A FILE TREE
        elif function["local"] == 0:
            self.basePackage = ""
            self.packageDir = ""
            if "." in function["package"]:
                self.basePackage += function["package"][0:function["package"].index(".")]
                self.packageDir += function["package"][(function["package"].index(".") + 1):]
            else:
                self.basePackage = function["package"]

            if "version" in function:
                self.version = function["version"]
            else:
                pkg = importlib.import_module(self.basePackage)
                if pkg:
                    self.version = importlib.metadata.version(self.basePackage)
                else:
                    warnings.warn("Couldn't find a version for the function being called: setting version to 0.0.0", UserWarning)
                    self.version = "0.0.0"
        # grab all the defaults of the particular function that aren't included in args
            
            pkg = importlib.import_module(self.basePackage + "." + self.packageDir)
            f = getattr(pkg, self.name)
            
            if f.__defaults__==None:
                full_args = {}
            else:
                full_args = dict(zip(f.__code__.co_varnames[-len(f.__defaults__):], f.__defaults__))
            print(self.name)
            print(full_args)
            print(pkg)
            shared_keys = tuple(full_args.keys() and args.keys())
          
        # change the default args to the user specified ones
        if shared_keys:
            for key in shared_keys:
                full_args[key] = args[key]
        else:
            full_args = args
        self.args = full_args
        
        self.internal = inter
        
        self.json = {"name": name, "description": description,"function": function, "channels": channels, "args": self.args, "inter": inter} 
    def __call__(self, data, ret=True):
        """The operator can be used to operate on a datum by specifying the data layers and channels. 

        The data will be stored in a strictly ordered vector which will be passed as a tuple to the function which defines the operator with the order being inherited from the order used to specify the channels. The result is stored in a single channel in a layer specified by the `channelOut` field."""

        if self.packageDir == "":
            pkg = importlib.import_module(self.basePackage)
        else:
            pkg = importlib.import_module(self.basePackage + "." + self.packageDir)
        f = getattr(pkg, self.name)
        
        if type(data) == Datum:
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
                    res = f(*v, **self.args)
            else:
                if self.args["unnamed"]:
                    v = self.args["unnamed"]
                    res = f(*d, *v, **self.args)
                else:    
                    res = f(*d, **self.args)
            
            for c in self.channelOut:
                if c == "preData":
                    data.preData[self.channelOut[c]] = res
                if c == "postData":
                    if inter["split"]:
                        for i in range(len(res)):
                            data.postData[self.channelOut[c] + str(i) ] = res[i]
                    else:
                        data.postData[self.channelOut[c]] = res
        
        elif type(data) == str:
            print(self.__dict__)
            f(data, **self.args)
        
        elif type(data) == DataSet: 
            for i in self.channelsIn:
                d = []
                if i == "preProcess":
                    if "totalAnalysisPreProcessing" in self.internal and self.internal["totalAnalysisPreProcessing"]:
                        d = data.data
                    else:
                        d = [data]
  
                elif i == "postProcess":
                    d = [[x.postData[g] for g in self.channelsIn[i]] for x in data.data]

                elif i == "analysis":
                    ### TO DO: ensure this works for a single data set as well as a list of datasets (recommended)
                    if "postAnalysis" in self.channelOut:
                        if len(self.channelsIn[i]) == 1:
                            d = [j.analysis[g] for g in self.channelsIn[i] for j in data.data]
                        else:
                            d = [[j.analysis[g] for g in self.channelsIn[i]] for j in data.data]
                    else:
                        d = [data.analysis[g] for g in self.channelsIn[i]]
                elif i == "postAnalysis":
                    if self.internal["totalDataSet"]:
                        d = [data]
                    else:
                        d = data.data
                else:
                    raise NameError("There is no valid layer by that name.")
            if "broadcast" in self.internal:
                # always broadcast over the first provided channel
                res = []
                for i in range(len(d[0])):
                    v = [x[i % len(x)] for x in d]
                    res.append(f(*v, **self.args))
            else:
                if "unnamed" in self.args:
                    v = self.args["unnamed"]
                    named = self.args.copy()
                    named.pop("unnamed", 'None')
                    if d:
                        res = f(*d, *v, **named)
                    else:
                        res = f(*v, **named)
                else:
                    res = f(*d, **self.args)
            
            for c in self.channelOut:
                if "split" in self.internal and self.internal["split"] == True:
                    if "broadcast" in self.internal:
                        step = len(res[0])
                        for i in range(step):
                            data.analysis[self.channelOut[c][0] + str(i)] = [r[i] for r in res]
                    else:
                        for i in range(len(res)):
                            data.analysis[self.channelOut[c][0] + str(i)] = res[i]
                else:
                    assert len(self.channelOut[c]) <= 1, "Multichannel output not yet supported"
                    
                    if self.channelOut[c] != []: # empty channels should not through errors or write results
                        data.analysis[self.channelOut[c][0]] = res

            if ret:
                return data


# Functions

# # # Piping
def _preprocess(recipe, bidsdir):
    r"""
    Apply a preprocessing pipeline to a directory of data in the BIDS standard. Currently only command line based preprocessing pipes are supported. A user defined pipeline can be specified using a series of operators on a list of directories.
    
    Parameters
    ----------
    recipe : graphpype.pipe.recipe
    bidsdir : string
        The location of the bidsdir

    Returns
    -------
    None
    """
    
    assert "preProcess" in recipe.nodes, "You need to specify the preprocessing operations." 
    
    ops = recipe.nodes["preProcess"]
    for i in ops:
        if i.basePackage == "cmd":
            cmdStr =[k + v for (k, v) in i.args.items()][0].split()
            subprocess.run(cmdStr)
        else:
            # the bidsdir includes all names of the subject paths as elements of a vector; the first element of the vector is the root of the data directory
            i(bidsdir)
            # [f(bidsdir) for f in ops]
    ops[1](bidsdir)

def _process(ops, dataset, nthreads: int, pool=None):
    r""" 
    Process a series of operators over a dataset.

    Parameters
    ----------
    ops : list
        A list of graphpype.pipe.operator objects that operate on the data.
    dataset : graphpype.pipe.dataset
        The dataset to be processed.
    nthreads : int, optional
        Number of threads to distribute over.
    pool : object, optional
    
    Returns
    -------
    None
    """

    if nthreads > 1:
      #  import multiprocessing
      #  for f in ops:
      #     if f.name != 'plots': ## Change this to env["nonparallel"]
      #          lambdaF = lambda x: f(x)
      #          pool.map(f, dataset)
      #      else:
      #          [f(d) for d in dataset]

        from joblib import Parallel, delayed, cpu_count
        from joblib.externals.loky import set_loky_pickler
        set_loky_pickler('pickle')
        pool = Parallel(n_jobs=-1, prefer='threads', backend='multiprocessing')
        for f in ops:
            if f.name != 'plots': ## Change this to env["nonparallel"]
                dataset = pool(delayed(f)(d, ret=True) for d in dataset)
            else:
                [f(d) for d in dataset]
        return dataset
    else:
        for d in dataset:            
            [f(d, ret=False) for f in ops]


            

