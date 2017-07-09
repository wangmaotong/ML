#将文本记录转换成NumPy的解析程序
from numpy import zeros


def file2matrix(filename):

    fr = open(filename)

    arrayOLines = fr.readlines()

    numberOfLines = len(arrayOLines)

    returnMat = zeros((numberOfLines,3))

    classLabelVector = []

    index = 0

    for line in arrayOLines:

        line = line.strip()

        listFormLine = line.split('\t')

        returnMat[index,:] = listFormLine[0:3]

        classLabelVector.append(int(listFormLine[-1]))

        index += 1

    return returnMat,classLabelVector