#归一化特征值
from numpy import zeros, shape, tile


def autoNorm(dataSet):  #归一化特征值

    minVals = dataSet.min(0)

    maxVals = dataSet.max(0)

    ranges = maxVals - minVals

    normDataSet = zeros(shape(dataSet))

    m = dataSet.shape[0]

    normDataSet = dataSet - tile(minVals,(m,1))

    normDataSet = normDataSet/tile(ranges,(m,1))

    return normDataSet,ranges,minVals