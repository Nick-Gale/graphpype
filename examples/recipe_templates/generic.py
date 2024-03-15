# Import dependencies

import numpy, networkx, scipy, matplotlib, graphpype


#----------------------------------------------------------------------------------------------------------
# Load the data
#----------------------------------------------------------------------------------------------------------

oLoadParcel = graphpype.pipe.Operator(
    name="Cortical Thickness Loader",
    description="Load cortical thickness data parcellated at 308 nodes for the three datasets",
    function={
            "name": "loadParcellation",
            "package": "graphpype.utils",
            "local": 0,
            "version": "0.0.1"
            },
    channels={
            "dataIndex": {"preProcess": []},
            "resultIndex": {"analysis": []}
            },
    args={"dataDirectory": "./data/ct.npy", "listFilterDirectory": "./data/group.txt", "channel": "cortical", "nameFilter": True},
    inter={"totalAnalysisPreProcessing": True}
)

oLoadCoords = graphpype.pipe.Operator(
    name="Coordinates Loader",
    description="Load Coordinates from the CSV file.",
    function={
        "name": "loadAnalysisChannel",
        "package": "graphpype.utils", "version": "0.0.1", "local": 0
    },
    channels={
        "dataIndex": {"preProcess": []},
        "resultIndex": {"analysis": []}
    },
    args={
        "dataDirectory": "./data/centroids_500.csv",
        "dataType": "csv",
        "channel": "distances"
        },
    inter={"totalAnalysisPreProcessing": True}
    ) # load the euclidean coordinates for each parcellation

#----------------------------------------------------------------------------------------------------------
# Do the internal group analysis
#----------------------------------------------------------------------------------------------------------

oPostCovariance = graphpype.pipe.Operator(
    name="Group Covariance Matrix",
    description="Each group has a particular cortical thickness dataset asscociated with it given as a NxL matrix of N subjects and L parcellations. The group covariance is calculated as LxL for the normalised covariance of all N subjects at a given parcellation location.",
    function={
        "name": "covarianceMatrix",
        "package": "graphpype.stats", "local": 0, "version": "0.0.1",
        },
    channels={
        "dataIndex": {"preProcess": ["cortical"]},

        "resultIndex": {"analysis": ["covariance"]}
        },
    args={}
    ) # calculate Covariances

oPostThickness = graphpype.pipe.Operator(
    name="Cortical Thicknesses",
    description="",
    function={
        "name": "loadFeature",
        "package": "graphpype.stats", "local": 0, "version": "0.0.1"
        },
    channels={
        "dataIndex": {"preProcess": ["cortical"]},
        "resultIndex": {"analysis": ["thickness"]}
        },
    args={}
    )

oPostGraph = graphpype.pipe.Operator(
    name="Groupwise Graph",
    description="Create the groupwise graph objects",
    function={
        "name": "constructMinSpanDensity",
        "package": "graphpype.graph", "local": 0, "version": "0.0.1"
        },
    channels={
        "dataIndex": {"analysis": ["covariance"]},
        "resultIndex": {"analysis": ["graph"]}
        },
    args={
        "density": 0.1
        }
    ) # calculate Graph

oPostGraphDistribution = graphpype.pipe.Operator(
    name="Graph Distribution",
    description="Permuting the cortical thicknesses to construct the covariance graph will result in a distribution of possible such graphs; unpermuting allows them to be comparable",
    function={
        "name": "constructedDensityPermutationGraph",
        "package": "graphpype.graph", "local": 0, "version": "0.0.1"
    },
    channels={
        "dataIndex": {"analysis": ["covariance"]},
        "resultIndex": {"analysis": ["randomGraph"]}
    },
    args={
        "density": 0.1,
        "nPermutations": 100,
        "seed": 0
        }
    )

oPostDegreeCDF = graphpype.pipe.Operator(
    name="Group Degree Distributions",
    description="Each group has a singular adjancency matrix constructed from composition of multiple data points. The degree distribution is calculated on this aggregate graph.",
    function={
        "name": "degreeCDF",
        "package": "graphpype.graph", "local": 0, "version": "0.0.1"
        },
    channels={
        "dataIndex": {"analysis": ["graph"]}, 
        "resultIndex": {"analysis": ["degreeDistribution"]}
        },
    args = {}
    ) # calculate degree distributions

oPostDistDegreeCDF = graphpype.pipe.Operator(
    name="Group Degree Distributions",
    description="Each group has a singular adjancency matrix constructed from composition of multiple data points. The degree distribution is calculated on this aggregate graph.",
    function={
        "name": "degreeCDF",
        "package": "graphpype.graph", "local": 0, "version": "0.0.1"
        },
    channels={
        "dataIndex": {"analysis": ["randomGraph"]}, 
        "resultIndex": {"analysis": ["distDegreeDistribution"]},
        },
    args={},
    inter={"broadcast": True, "split": True}
    )

oPostDegreeCorticalThickness = graphpype.pipe.Operator(
    name="Graph Cortical Thickness",
    description="",
    function={
        "name": "featureDegreeDistribution",
        "package": "graphpype.graph", "local": 0, "version": "0.0.1"
        },
    channels={
        "dataIndex": {"analysis": ["graph", "thickness"]},
        "resultIndex": {"analysis": ["thicknessDegreeDist"]}
        },
    args={},
    inter={"split": True}
    )


oPostLouvain = graphpype.pipe.Operator(
    name="Group Louvain Clustering",
    description="Calculates the community structure for each group using Louvain clustering on the groupwise graph.",
    function={
        "name": "louvainCommunities",
        "package": "graphpype.graph", "local": 0, "version": "0.0.1"
        },
    channels={
        "dataIndex": {"analysis": ["graph"]}, 
        "resultIndex": {"analysis": ["louvain"]}
        },
    args={
        "seed": 0
        }
    ) # Louvain-Clustering

oPostGenerateRandomCommunity = graphpype.pipe.Operator(
    name="Generate Random Communities",
    description="Using a community partition as a seed generate random graphs with a similar structure according to the Stochastic Block Model",
    function={
        "name": "randomCommunityStochasticBlock",
        "package": "graphpype.graph", "local": 0, "version": "0.0.1"
        },
    channels={
        "dataIndex": {"analysis": ["graph", "louvain"]},
        "resultIndex": {"analysis": ["randomCommunity"]}
        },
    args={
        "nGraphs": 100,
        "seed": 0
        }
    ) # append Permutations

oPostConnectedDistance = graphpype.pipe.Operator(
    name="Connected Distance Distributions",
    description="",
    function={
        "name": "estimateDistancePermutation",
        "package": "graphpype.stats", "local": 0, "version": "0.0.1"
        },
    channels={
        "dataIndex": {"analysis": ["graph", "distances"]},
        "resultIndex": {"analysis": ["connectedDistDistribution"]}
        },
    args={}
    
) # connected distance distributions

oPostD2 = graphpype.pipe.Operator(
    name="SquaredDistances",
    description="",
    function={
        "name": "distanceMat",
        "package": "graphpype.utils", "local": 0, "version": "0.0.1"
        },
    channels={
        "dataIndex": {"analysis": ["distances"]},
        "resultIndex": {"analysis": ["distanceMat"]}
        },
    args={}
    ) 
#----------------------------------------------------------------------------------------------------------
# Do the comparative group analysis
#----------------------------------------------------------------------------------------------------------

oAnalysisSignificanceDegree = graphpype.pipe.Operator(
        description="There are 1000 possible degree distributions generated by permutations of the covariance matrix. For each degree level we test for signficance between the populations and correct using the Family Wise Error Rate",
        function={
            "name": "compareGroupDegreeMeans",
            "package": "graphpype.stats", "local": 0, "version": "0.0.1"
            },
        channels={
            "dataIndex": {"analysis": ["distDegreeDistribution1"]},
            "resultIndex": {"postAnalysis": ["significantDegrees"]}
            },
        args={"threshold": 0.025}
        ) # Compare Degree Distributions

oAnalysisSignificanceThickness = graphpype.pipe.Operator(
        description="Calculate the signficance levels for the thickness at the degree distribution level",
        function={
            "name": "multipleTTest",
            "package": "graphpype.stats", "local": 0, "version": "0.0.1"
            },
        channels={
            "dataIndex": {"analysis": ["thicknessDegreeDist0", "thicknessDegreeDist1", "thicknessDegreeDist2"]},
            "resultIndex": {"postAnalysis": ["significantThickness"]}
            },
        args={"threshold": 0.025}
        ) # Compare Degree Distributions


oAnalysisDistancePearsons = graphpype.pipe.Operator(
    name="Distance Pearsons Relationship",
    description="Each group wise matrix of correlations is compared to the Euclidean distances between parcellations",
    function={
        "name": "generalLinearModel",
        "package": "graphpype.stats", "local": 0, "version": "0.0.1"
        },
    channels={
        "dataIndex": {"analysis": ["covariance", "distanceMat"]},
        "resultIndex": {"analysis": ["r2distanceFit"]}
    },
    args={"sets": []}
    ) # Pearsons Correlation as Function Euclidean Distance


oModularOverlap = graphpype.pipe.Operator(
    name="Modular Overlap",
    description="For each pair of nodes in each module in a partition check if the nodes share a module in the corresponding partition which can be a different grouping or a random parition from a null-model. For all possible pairs (note: not all possible pairs of nodes in the graph unless there is only one module) compute this membership and return the fraction of pairs which share modules in partitions. The z-score of the pairing (e.g. ADHD-Autism) is computed assuming the distribution of the null model is normal and representative",
    function={
        "name": "modularZTest",
        "package": "graphpype.stats", "local": 0, "version": "0.0.1"
        },
    channels={
        "dataIndex": {"analysis": ["louvain", "randomCommunity"]},
         "resultIndex": {"postAnalysis": ["modularOverlap"]}},
    args={}
    ) # Modularity

oAnalysisModularDistributionTest = graphpype.pipe.Operator(
    name="Modular Overlap Differences",
    description="Test if the distributions of modular overlaps differences between pairs of modular overlaps are signficant given surrogate distribution partitions.",
    function={
        "name": "pairgroupModularZTest",
        "package": "graphpype.stats", "local": 0, "version": "0.0.1"},
    channels={
        "dataIndex": {"analysis": ["louvain", "randomCommunity"]},
        "resultIndex": {"postAnalysis": ["groupModularOverlap"]}
        },
    args={}
    )
#----------------------------------------------------------------------------------------------------------
# Do the plots and outputs 
#----------------------------------------------------------------------------------------------------------
 
oPlots = graphpype.pipe.Operator(
    name="Plotting",
    description="Generate the plotting objects in the plotting file.",
    function={
        "name": "plots",
        "package": "graphpype.utils", "local": 0, "version": "0.0.1"},
    channels={
        "dataIndex": {"postAnalysis": []},
        "resultIndex": {"postAnalysis": ["plots"]}
    },
    args={
         "plotsDir": './plots.py'
        },
    inter={
        "totalDataSet": True
        }
  )



#----------------------------------------------------------------------------------------------------------
# Run the recipes
#----------------------------------------------------------------------------------------------------------

# Top level recipe information

name = "Reproduction study of the paper: Structural Covariance Networks in Children with Autism or ADHD"

descr = "This recipe attempts to reproduce the papers graph theoretical analysis. The preprocessing steps are assumed to be conducted flawlessly and there is no reference to individual data points, only group-wise adjacency matrices are considered. Unless otherwise stated all statistical tests are assumed to be a derivative of the General Linear Model and errors are thus distributed normally."

env = {"nThreads": 3, "seed": 1}

nodes = {
            "preProcess": [oLoadParcel, oLoadCoords],
            "postProcess": [oPostCovariance, oPostThickness],
            "analysis": [oPostD2, oPostGraph, oPostGraphDistribution, oPostDegreeCDF, oPostDistDegreeCDF, oPostLouvain, oPostGenerateRandomCommunity, oPostDegreeCorticalThickness, oPostConnectedDistance, oAnalysisDistancePearsons],
            "postAnalysis": [oAnalysisSignificanceDegree, oModularOverlap, oAnalysisModularDistributionTest, oAnalysisSignificanceThickness, oPlots]
        }

recipe = graphpype.pipe.Recipe(
        name,
        descr,
        nodes,
        env
        )
