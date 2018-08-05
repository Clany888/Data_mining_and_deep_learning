import numpy as np
import operator

"""
计算已知类别数据集中的点与当前点之间的距离；
按照距离递增次序排序；
选取与当前点距离最小的k个点；
确定前k个点所在类别的出现频率；
返回前k个点所出现频率最高的类别作为当前点的预测分类。
"""
def createDataSet():
    """
    函数说明 : 创建数据集
    Parameters:
    无
    Returns:
        group - 数据集
        labels -分类标签
    """
    #四组二维特征数值
    group = np.array([[1,101],
                      [5,89],
                      [108,5],
                      [115,8]])
    #四组特征的标签
    labels = ['爱情片','爱情片','动作片','动作片']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
    函数说明:kNN算法,分类器

    Parameters:
        inX - 用于分类的数据(测试集)
        dataSet - 用于训练的数据(训练集)
        labes - 分类标签
        k - kNN算法参数,选择距离最小的k个点
    Returns:
        sortedClassCount[0][0] - 分类结果
    """

    # numpy函数shape[0]返回dataSet的行数   4
    dataSetSize = dataSet.shape[0]

    """
    tile:比如 a = np.array([0,1,2]),   
    np.tile(a,(2,1)) 行复制2倍，列复制一倍
    array([[0,1,2], 
          [0,1,2]])
    """
    # tile方法 在列向量方向上重复inX共1次(横向)，行向量方向上重复inX共dataSetSize次(纵向)
    #   np.tile(inX, (dataSetSize, 1))=
         # [[101,20],
           # [101,20],
           # [101,20],
           # [101,20]]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet

    # 二维特征相减后平方
    sqDiffMat = diffMat ** 2

    # sum()所有元素相加，sum(0)列相加，sum(1)行相加
    sqDistances = sqDiffMat.sum(axis=1)

    # 开方，计算出距离
    distances = sqDistances ** 0.5

    # 返回distances中元素从小到大排序后的索引值,得到的是一个列表
    sortedDistIndices = distances.argsort()
    # argsort() 返回的是原始裂变排序好的索引值

    # 定一个记录类别次数的字典
    classCount = {}
    for i in range(k):
        # 通过遍历取出前k个元素的类别
        voteIlabel = labels[sortedDistIndices[i]]
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        # 计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # python3中用items()替换python2中的iteritems()
    # items() 字典的items方法作用：是可以将字典中的所有项，以列表方式返回。可以用于 for 来循环遍历。
    # 字典的iteritems方法作用：与items方法相比作用大致相同，只是它的返回值不是列表，而是一个迭代器。
    # key=operator.itemgetter(1)根据字典的 值 进行排序
    # key=operator.itemgetter(0)根据字典的 键 进行排序
    # reverse降序排序字典  第一个即为最终的分类结果
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    """
    这里通过统计类别出现的次数其去代替类别所占概率，效果是一样的。
    """
    # 返回次数最多的类别,即所要分类的类别，即预测出了所给的测试电影的种类。
    return sortedClassCount[0][0]

if __name__ == '__main__':
    # 创建数据集
    group, labels = createDataSet()
    print(group,labels)
    # 测试集
    test = [2, 100]
    # kNN分类
    test_class = classify0(test, group, labels, 2)
    print('---------------------')
    print(test, group)
    print('---------------------')

    # 打印分类结果
    print(test_class)