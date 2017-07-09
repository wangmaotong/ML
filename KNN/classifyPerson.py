#约会网站预测函数

from numpy.ma import array

from KNN.auroNorm import autoNorm
from KNN.classify0 import classify0
from KNN.file2matrix import file2matrix


def classifyPerson():

    resultList = ['not at all','in small doses','in large doses']

    percentTats = float(input('percentage of time spent playing video games?'))

    ffMiles = float(input('frequent flier miles earned per year?'))

    iceCream = float(input('liters of iceCream consumed per year?'))

    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')

    norMat,ranges,minVals = autoNorm(datingDataMat)

    inArr = array([ffMiles,percentTats,iceCream])

    classifierResult = classify0((inArr - minVals)/ranges,norMat,datingLabels,3)

    print('You will probably like this Person: ',resultList[classifierResult - 1])

classifyPerson()
