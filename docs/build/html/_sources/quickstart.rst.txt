QuickStart
===============

Graphpype aims to create reproducible and shareable pipelines for analysis in neural imaging. We aim to do this by introducing the concept of a recipe. The recipe is a shareable format which specificies how a particular analysis pipeline is to be run. 

=======
Recipes
=======
The recipe is composed of several fields each specifying the metadata of a particular aspect of the pipeline: `name`, `description`, `nodes`, `env`, and optionally `template`. A recipe is stored in a '.json' format for portability and may be specified as such, but it is easier to build in a python session or python executable script and will result in less errors. We can start by initialising an empty recipe.

.. code-block:: python

    import graphpype
    emptyRecipe = graphpype.pipe.Recipe()

The `name` and `description` fields are strings and are intended to be shorthand descriptors about the recipe for the end user. These can specified by keyword arguments or they can be built constructively.

.. code-block:: python

   namedRecipe = graphpype.pipe.Recipe(name="An empty recipe", description="A fairly useless description")
   emptyRecipe.name = "Another empty recipe"

The `nodes` field is a dictionary that contains the nodes of the graph that defines the pipeline. These are ordered into levels of analysis that are strictly ordered as: 'preProcess', 'postProcess', 'analysis', and 'postAnalysis'. Each of these contains a list of `Operators` which are a graphpype class that defines an operation on a (possibly empty) set of  data. The list is read with left-to-right ordering which allows nodes on the same level to inter-operate. While rather strict, this formulation allows for a natural and portable expression of any generic analysis pipeline. Furthermore, expressing the pipeline as a graph opens the pipeline to analysis itself e.g. automatic differentiation through parameters would allow for sensitivity analysis which currently is not mainstream practice in neuroimaging studies.

The recipe is a thus a coherent organisation of these operators that allow an analysis pipeline to transform raw data into a result: statistic, plot, or otherwise. These specifications are completely generic to any scientific analysis but the levels are chosen to closely align with neuroimaging. The preProcess level refers to individual raw data, the postProcess data refers to individual data that has gone through a processing to say, a graph, the analysis level which collates the results generated over a particular dataset, and postAnalysis which includes analysis between multiple datasets, plots, and output operators. These concepts of data will be formalised in Classes that graphpype provides in the following section.

=========
Operators
=========
The `Operator` class itself is quite complicated and defines a lot of information with arguments: `name`, `description`, `function`, `channels`, `args`, and `inter`. The name and description follow from the recipe and allow quick summaries of what each operator does. The `function` is a dictionary which specifies the name of the function, the package directory where the function is located (this could be a local module in which we encourage the local module to be the same directory as the recipe), local (a binary indicating whether the code is stored locally), and version (which is unecessary in the case of an installed package). The `args` is an optional dictionary of arguments that are passed to the function - it is key to note that if all arguments are not provided `graphpype` will rely on the defaults and will record these in the '.json' specification. The `channels` is a dictionary specifiying the input and ouput locations of the data: these are specified by keys on a particular analysis level. The keys are "dataIndex" and "resultIndex". These each contain a dictionary arranged into levels and a list of channels for each level which the operator will accept as input. The result index has only one channel. Finally, `inter` specifies internal arguments which are hyperparameters: for example specifying `broadcast` will broadcast the operator over the first data index rather than accepting a whole list. This can be useful for example if you have a list of vectors and want to find the mean of each of them, `numpy.mean` will return the global mean of the vector rather than returning the mean of each indivdiual vector. 

This method of specifying functions is labourious and can seem an unecessarily verbose way of specifying a simple function call. The advantage provided here is in the reproducibility: `graphpype` allows the correct version to be selected, the parameters to be front-ended (rather than buried or split between multiple scripts), and the whole recipe to be visualised as a graph allowing data dependencies to be exposed. The standardisation also allows for minimal changes to be made while building, debugging, and reproducing other work. Let's specify a simple operator.

.. code-block:: python

   import numpy
   import graphpype

   oMean = graphpype.pipe.Operator(
                name="mean",
                description="Takes the mean of the wiring distance",
                function={name: "mean",
                          package: "numpy",
                          local: 0
                          }
                channels={dataIndex: {postProcess: "wiringDistance"},
                          resultIndex: {analysis: "cost"}
                          }
                args={},
                inter={}
                )

This is a simple operator that doesn't require any input arguments or internal hyperparameters and at a glance appears to use the mean function in the `numpy` package to transform the wiring distance of a graph to a simple cost measure. When operated on the version of numpy recorded will be used and any unknown parameters e.g. `axis`. Then, if there was a conflict between recipes on different computers it might be resolved by comparing numpy versions and checking their releases for particular bugs. Also, if someone was reproducing a paper and didn't realise that `axis=1` should be specified the json template would front-end the default specification of `None` which would aid in resolving the conflict.

============================
Constructing Example Recipes
============================

There are some standard operations that one might want to apply to their analysis pipeline. To help building a recipe we offer two methods: the template keyword and a function in the `utils` package.

.. code-block:: python

   import graphpype
   recipeStandard = graphpype.pipe.Recipe(template="standard")
   graphpype.utils.generateTemplate(name="generic", exampleFile="generic.py")
   recipeRead = recipeStandard.read("path/to/recipe.json")

The first method reads out from templates that are bundled within the graphpype package. The second method allows for a more abritary specification that immediately writes the recipe specified in a python script. Finally, the Recipe class provides a method `read` which accepts a directory from which a '.json' file may be read from. Once we are happy with our recipe we may write it to disk using the `write` method.

.. code-block:: python

   recipeStandard.write("path/to/recipe.json")

The resulting file now defines a portable recipe template. We might also like to get an idea of what our recipe actually does. A useful visualisation of the recipe is given by `generateFlowchart` in the `utils` package which will generate a visualation of the graph defining the recipe. There are default colours provided for each level of the analysis but these can be specified as a dictionary of named colours for customisation. If the data paths are known we can specify their name and paths as dictionary key values and graphpype provides the `report` method which will write a report as a html file summarising the data and operator chain. We have a data directory with two BIDS formatted subdirectory which we will label as "A" and "B". Let's generate a report card and a flowchart for this recipe and dataset.

.. code-block:: python
   
   import graphpype
   recipeStandard = graphpype.pipe.Recipe(template="standard")
   recipeStandard.write("ourNewExample")
   plotObj = graphpype.utils.generateFlowchart(recipeStandard)
   plotObj.show()
   bidsDict = {"A": "data/splitA", "B": "data/splitB"} 
   recipeStandard.report(recipe, bidsDict, outputDir="./data/derivatives/", author="Your Name")

For some more examples of recipe templates refer to the Examples page in the documentation.

=========
Pipelines
=========

A recipe is a generic graph specifying the transformations of data and datasets toward a final result. A pipeline is the application of a recipe to a specific dataset or set of datasets. The `Pipeline` class is provided to construct and process analysis pipelines on a particular compute environment. There is an internal representation of data that the `Pipeline` class handles in the form of the `Datum` and `Dataset` classes.

.. code-block:: python
   
   import graphpype
   pipe = graphpype.recipe.Pipeline('./ourNewExample', bids=bidsDict, preprocess=True) 

==================
Datum and Datasets
==================

The `Datum` class is the atomic unit of the data and has three fields: dirs, preData, and postData. The dirs specifies the directory where raw data is kept. The preData refers to the channels of assigned to the preProcessing level. The postData analagously refers to the channels assigned to the postProcessing level. As an example, a Datum might refer to a subject which has had an fmri scan and DTI scan and has had this data stored in a BIDS directory under 'subj-0001'. Some preprocessing might have been applied to the data to generate a FMRI time-series, and some post-processing on this preprocessed data to generate a covariance matrix and graphs.

The `DataSet` class refers to a collection of either atomic units `Datum` or of `DataSets`. The fields are `name`, `data`, and `analysis`. The name field is for specification useful mainly when multiple datasets are used. The data field is simply a list of pointers to Datum objects, or Datasets. The `analysis` field is a dictionary which contains the results of operators which export to the analysis or postAnalysis levels.

For the most part graphpype will automatically generate the Datum and Datasets typically through an operator. It is possible to individually manipulate the data but it is not recommended outside of exploratory analysis. We can specify the data objects by providing the BIDS directories.

===========
Parallelism
===========

Parallelism is internally handled in graphpype explicitly over lists of Datum or Dataset objects. There are some operators which have native graphpype implementations that utilise GPU acceleration and there is no restriction of parallelism (e.g. Freesurfer) from calls from other packages. The `nthreads` internal argument specifies the number of threads that can be used and assigns a dataset/datum to each thread and runs the operator over that thread before collection.

==========
Example
==========

Let's now crystalise this in a runnable example with our dataset of A and B. This assumes that you have `graphpype`, the relevant dependencies for fmri analysis, and python packages installed. It also assumes a data folder with the following structure:

    * ./data/
       - derivatives/
       - splitA/
         -derivatives/
       - splitB/
         -derivatives/

The following code block pulls a recipe template from the available options, writes the recipe in JSON format to the top level folder of your analysis, generates a HTML report summarising the recipe and a flowchart defining the analysis graph, and finally processes the pipeline of the recipe distributed over the two independent datasets labelled "A" and "B" and found in the "splitA" and "splitB" data directories.


.. code-block:: python
   
   import graphpype

   recipeStandard = graphpype.pipe.Recipe(template="standard")
   recipeStandard.write("ourNewExample")
   plotObj = graphpype.utils.generateFlowchart(recipeStandard)
   plotObj.show()
   bidsDict = {"A": "data/splitA", "B": "data/splitB"} 
   recipeStandard.report(recipe, bidsDict, outputDir="./data/derivatives/", author="Your Name")
   pipe = graphpype.recipe.Pipeline('./ourNewExample', bids=bidsDict, preprocess=True) 
   pipe.process()

