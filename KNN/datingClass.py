#测试代码


from KNN.auroNorm import autoNorm
from KNN.classify0 import classify0
from KNN.file2matrix import file2matrix

def datingClass():
    hoRatio = 0.1
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResul = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with : %d,the real answer is: %d" %(classifierResul,datingLabels[i]))
        if(classifierResul != datingLabels[i]):
            errorCount += 1.0
    print('the total error rate is: %d'%(errorCount/float(numTestVecs)))

datingClass()