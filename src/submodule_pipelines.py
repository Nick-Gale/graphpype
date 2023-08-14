# Submodule: pipelines. Functions dedicated to constructing pipelines: pre-processing, processing, plotting, and end-to-end functionality.

# Constants

# Classes

class Recipe:
    """Pipeline Recipe: a directed line graph of functions/constants that operate on a subset of data for analysis, output, and plotting. The recipe informs the operations that must be sequentially applied to data and subsequent output manipulations. Recipes can be composed by concatentation."""

    name: str
    """Recipe identifier"""

    description: str
    """Summary of the pipeline flow i.e. brief description of what the recipe cooks."""

    nodes: list["Operator"]
    """List of functions that the analysis will operate with"""

    seed: int
    """The seed used for all randomisation tasks. A recipe is considered incomplete without a seed as it is required for reproducibility. Default value: 1."""
    
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


class Operator:
    """A processing operator that operates on a discrete chunk of data which either is broadcastover/reduces a data set"""

    # TO DO


# Functions

# # # Piping

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

def _cache_data_pipe(recipe, data, cache_directory)
    
    # if exists(cache_directory) load else create cache
    if:
        cache = [data]
    else:
        cache_number = _check_cache(recipe, cache_directory)
        
        assert cache_number > 0
        
        cache = load_cache(recipe, cache_directory, cache_number) 

    result = _cache(recipe, dataset, cache, cache_directory)
    return result

# # # Caching
def _cache(recipe, dataset, cache, cache_directory):
    """Cache the outputs of a recipe applied to a dataset. Can be used to skip uneccessary computational analysis at the expense of memory."""

    for i in cache_number:(len(data.nodes)-1):
        result.append(data.nodes[i](result[i-1]))
    
def _check_cache(recipe, cache_directory):
    """Finds the first node that is divergent between recipes."""

def _load_cache(recipe, cache_file, cache_number):
    """Loads pre-cached output until the recipes diverge."""


