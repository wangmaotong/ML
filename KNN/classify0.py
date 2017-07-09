#K近邻算法


from numpy import tile
import operator


def classify0(inX,dataSet,labels,k):   #inX:用于分类的输入向量，dataSet：训练样本的集合，labels：标签向量，k：K-近邻算法中的k

    dataSetSize = dataSet.shape[0]  #确定矩阵有多少行，同理shape［1］确定矩阵有多少列

    diffMat = tile(inX,(dataSetSize,1)) - dataSet

    sqDiffMat = diffMat**2

    sqDistances = sqDiffMat.sum(axis=1)

    distances = sqDistances**0.5

    sortedDistIndicies = distances.argsort()

    classCount = {}

    for i in range(k):

        voteIlable = labels[sortedDistIndicies[i]]

        classCount[voteIlable] = classCount.get(voteIlable,0) + 1

        sortedClassCount = sorted(classCount.items(), key =operator.itemgetter(1),reverse=True)

        return sortedClassCount[0][0]