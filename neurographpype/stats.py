import scipy
import numpy
import statsmodels

def compareDegreeDist():
    """Assesses the degree to which two empirical distributions are statistically different. There are some common ways of doing this: assuming the degree distribution takes a specific functional form, or the general Kolmogorov-Smirnov test. Support is provided for degree distributions assumed to be in the power-law form or the Kolmogorov-Smirnov through the `test` keyword  (default `test='Kolmogorov-Smirnov'`)"""

    # assert type is list of integers, or two ecdfs

    # case switch KS or powerlaw
    case test=='Kolmogorov-Smirnov':
        return scipy.stats.ktest(dist1, dist2)
    case test=='power':
        power
    # fit

    # report P-value
    return None



def generatePermutations(datum, n, method='degree'):
    """Generate a random set of permutations for a given adjacency matrix that serve as a null-model distribution under some assumption i.e. preserving degree distribution. Currently supported options are: 'degree', 'spin', 'random'. Degree distributions are generated using the FILLTHISIN method from FILLTHISINPACKAGE/PAPER. Spin models follow the structure presented by Block et. al. and implemented in FILLTHISINPACKAGE. Random assumes no structure and provides random permutations of the matrix."""
    
    A = datum.adjacency

    L = A.shape[0] 

    case method='degree':

    case method='random':

    case method='spin':

    return None

def covarianceMatrix(data, normalise=True):
    """Returns the covariance matrix (size: L x L) of a data distributed amongst a particular parcellation (size: L) for a particular dataset (size: N). The default behavior is to normalise by standard deviations returning the regular Pearsons correlation coefficient."""
    L = len(data[0]) # the dimension of the parcellation
    y = len(data[0][0]) # the dimension of the data e.g. single estimate or a time series

    assert all([len(data[0]) == len(data[i]) for i in range(len(data))), "All covariance comparisons should be of the same data dimensionality."

    dataMat = numpy.array(data)
    mat = numpy.zeros(L, L)
    for i in range(L):
        for j in range(L):
            if y == 1:
                if normalise:
                    mat[i, j] = scipy.stats.pearsonsr(dataMat[:, i], dataMat[:, j])[0]
                else:
                    mat[i, j] = numpy.cov(dataMat[:, i], dataMat[:, j])[0][1]
            else:
                vi = utils.vectorSlice(dataMat, i, 2)
                vj = utils.vectorSlice(dataMat, j, 2)
                if normalise:
                    mat[i, j] = scipy.stats.pearsonsr(vi, vj)[0]
                else:
                    mat[i, j] = numpy.cov(vi, vj)[0][1]

    return mat

def generalizedLinearModel(data, sets, covariateChannels, regressorChannels, link=None, test='none'):
    
    if sets = []:
        # regressing on a single data set

    else:
        # regressing on multiple datasets
        
        fit = {}     
        for x in range(len(sets)):
            fit[x] = {}
            for y in range(len(sets)) if y != x:
                
                # Load data by index    
                if type(sets[x]) == str:
                    index = where([sets[x] == i.name for i in data.data])
                    dataX = data.data[index]
                elif type(sets[x]) == int:
                    dataX = data.data[x]
                else:
                    raise("Please address the data by a named index, or an integer corresponding to the linear index of the dataset")

                if type(sets[y]) == str:
                    index = where([sets[y] == i.name for i in data.data])
                    dataY = data.data[index]
                elif type(sets[x]) == int:
                    dataY = data.data[y]
                else:
                    raise("Please address the data by a named index, or an integer corresponding to the linear index of the dataset")
            
                # Arrange data and fit
                case test='none':
                    regressors = numpy.array([getattr(d, c) for c in regressorChannels for d in dataX])
                    covariates = numpy.array([getattr(d, c) for c in covariateChannels for d in dataY])
                    if link=None:    
                        glm = statsmodels.OLS(regressors, covariates)
                    else:
                        linkfamily = getattr(statsmodels.families, link)
                        glm = statsmodels.GLM(regressors, covariates, family=linkfamily)
                    
                    fit[x][y]["model"] = glm

                case test='ttest':
                    assert dataX.shape[1] == dataY.shape[1], "The data must have the same dimension for a two sided t-test"
                    
                    # split into two cases for multiple data: either testing generic vector mean between two datasts, or lots of different categorical means within data set (needs correction)

                    prediction = numpy.concat([dataX, dataY], axis=1)
                    dummyMask = numpy.concat([numpy.zeros(dataX.shape[0]), numpy.ones(dataY.shape[0])], axis=0)
                    model = statsmodels.OLS(dummyMask, prediction)
                    fit[x][y]["testStastic"] = 
                    fit[x][y]["pValue"] = 
                    
                case test='anova':
                
                case test='ancova':

                case
    
