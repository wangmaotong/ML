#计算数据集的想弄熵

from math import log

def calcShannonEnt(dataSet):

    numEntries = len(dataSet)

    labelCounts = {}

    for featVec in dataSet:

        currentLabel = featVec[-1]

        if currentLabel not in labelCounts.keys():

            labelCounts[currentLabel] = 0

        labelCounts[currentLabel] += 1

    shannonEnt = 0.0

    for key in labelCounts:

        prob = float(labelCounts[key])/numEntries

        shannonEnt -= prob * log(prob,2)

    return shannonEnt



def createDataSet():

    dataSet = [[1,1,'maybe'],
               [1,1,'yes'],
               [1,0,'all'],
               [0,1,'haha'],
               [0,1,'no']]

    labels = ['no surfacing','flippers']

    return dataSet,labels

myDat,labels = createDataSet()
print(calcShannonEnt(myDat))