# Submodule: Data Preprocessing. A collection of functions that prepare raw data 

# Constants

# Classes 

class DataSet:
    """A data set class that a recipe can operate on. A list of data tuples which can represent multiple paired observations e.g. an fMRI scan and a gene regulatory network can form a tuple for a single specimen/patient."""
    
    name: str
    """Name of the dataset"""
    
    description: str
    """A short description of the dataset"""

    dtype: "DataType"
    """The datatype (i.e. (Float, ArrayDims(...))) for each element in the dataset."""

    data: list
    def __init__(self, name, description):
        """Initialises an empty data set"""

class DataType:
    """The type of the data assumed to a tensor with a given number of dimensions."""
    name: str
    description: str
    primitive
    dimensions: list[float]
    
    def __init__(self, data, n, d):
        self.name = n
        self.description = d
        self.dimensions = np.size(np.array(data))
        self.primitive = 


class Network: # <: DataType
    """A given network datum"""

# Functions

def _concatenate_raw(dirs...):
    """Take a list of data directories and concatenate"""


# Pre-Processing Functions
# ----------------------------------------------------------------------------------------
# NOTE: Need to import the NiPy libs here. Have some basic preprocessing data 
# NOTE: Need to import genetic-imaging preprocessing here. Are there standarised formats for this. What do we actually need from this modality.
# NOTE: Need to fully list all preprocessing dependencies. This is a difficult propsect due to lack of clarity. Strategy is to implement on demand for a different imaging modality
# NOTE: MRtrix support and other compiled programs. DTI will be a reference modality at some point - MRtrix is a CLI but compiled program for comprehensive preprocessing.
# ----------------------------------------------------------------------------------------
