import stan as st
import scipy.stats as stat
import numpy as np
import itertools as it
import statsmodels.api as sm

"""A series of utility functions related to computing various useful statistical measures. These functions assume data in the form of N x N x S tensors where the first and second indexes are the same dimension and the final dimension stores the samples."""
class ComparisonSet:
    """Set of all statistical tests performed between two datasets."""

    def __init__(self, dataset1, dataset2, recipe):
        




def _significance(pvals, siglevel, correction='bonferroni'):
    """Reporting significance levels after a list of p-values is supplied under a correction for Type I errors. """
    assert (correction=='bonferroni' or correction=='falsediscoveryrate' or correction=='uncorrected')
    nsamples = len(pvals)
    match correction:
        case: 'bonferroni':
            sig = pvals < siglevel / nsamples
        
        case: 'falsediscoveryrate':
            sig = pvals < siglevel * rankdata(pvals) / nsamples

        case: 'uncorrected':
            sig = p < siglevel

        case _:
            raise ValueError('Please specify a correction method.')

    return sig


def _power(pvals, nobs, test='ttest')
    """Reporting power levels of a series of independent test results under the assumptions of a specific test e.g. t-test."""


    return power

#--------------------------------------------
# Comparison Tests
#--------------------------------------------
# The data are in the form of graphs and can be split into control-test groups. A particular statistic vector is calculated as a function of the graph (e.g. average degree, or ROI corellation) across the dataset yielding a distribution of statistics. These can be compared between two groups to identify significant differences in a range of frequentists tests which can be assessed for statistical siginifance (Type I errors) and statistical power (Type II errors).

def glm(X, Y):
    """The General Linear Model regresses predictors X onto data Y and can be used as a generalisation of a broad class of statistical tests. It assumes errors are normally distributed; for a more general model where errors are assumed to be in the exponential family see Generalised Linear Model. Returns the fit alongside t-test and f-test results."""
    model = sm.OLS(Y, X)
    result = model.fit()

    return result.params, result.t_test, result_f_test

# def oneway_anova()

# def pairwise_binomial()

# def pairwise_manwhitney()



#--------------------------------------------
# Spatial Alignment Tests
#--------------------------------------------
# Comparing information between two maps (between patients, or between maps (fMRI to DTI) is an important question e.g. which certain regions implicated in a specific task? These questions are usually answered with a correspondence measure e.g. mutual information. There is a possibility that the spatial alignment measure can arise by chance. To answer the probability that an alignment/correspondence measure arises by chance a spin test is performed which provides a null model: the brain map is projected to the sphere and rotated before projecting back on to the brain surface and the correspondence measure is calculated, then with N measurements a distribution of the measure is formed and can be compared to an alternative map of interest to assess whether the maps are similar under measure because the particular spatial regions are important or if the measurement could have arisen by chance. https://github.com/spin-test/spin-test

def spatial_alignment(map1, map2, measure, nsamples, siglevel):

    # assert type(map1) == type(map2) and map1.shape == map2.shape

    # assert type(measure) == function 
    
    # dist = empericaldist ( [measure(map1 \circ r)) for r in genrots(nsamples)] )

    # pval = dist.cdf(measure(map2))

    # return pval < siglevel, pval


#--------------------------------------------
# Classifiers and Regressors
#--------------------------------------------

class TrainingProtocol:

    def __init__(self):


def _ml(dataset, permutation, split, classification_targets, model, training_protocol, seed=1):
    """Internal function to train a generic classifier. Generates a permutation of the data which is then split into test and training sets. The model is assumed to be an object with all hyperparameters encoded e.g. a neural network with predefined layers """
    


    return model, model_validation_outputs



