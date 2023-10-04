# Pseudocode Outline.

## Pipeline:
Level 0: Directories and Raw Data
Level 1: Preprocessed data; attached to the datum object with a channel reference for each operator.
Level 2: Processed data; attached to the datum object with a channel reference for each operator.
Level 3: Batch processing of the entire data set ( means, statistics, neural network training etc.); attached to the DataSet object with a channel reference for each operator.
Level 4: Comparison statistics between data sets ( condition1 v condition 2, training vs validation etc.); attached to the DataAnalysis Object with a channel reference for each operator.
Level 5: Output (recipe, plots, statistics tables); attached to the DataAnalysis Object.

Type Hierarchy:

Directory

|
V

_PreProcessing_

Datum / Data{
"""An indexed reference to a subject/specimen. This object contains a reference to the files used to generate the data
}

| ^
v |

_PostProcessing_

| ^
v |

DataSet{

}

|
V

_BatchProcessing_

|
V

