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

    data: list["Data"]
    def __init__(self, name, description):
        """Initialises an empty data set"""

class DataType:
    """The type of the data assumed to a tensor with a given number of dimensions."""

# Functions

def _concatenate_raw(dirs...):
    """Take a list of data directories and concatenate 
