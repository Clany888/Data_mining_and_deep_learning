from math import log
 
"""
函数说明:创建测试数据集
Parameters:
    无
Returns:
    dataSet - 数据集
    labels - 分类属性
"""
def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],       #数据集
                [0, 0, 0, 1, 'no'],
                [0, 1, 0, 1, 'yes'],
                [0, 1, 1, 0, 'yes'],
                [0, 0, 0, 0, 'no'],
                [1, 0, 0, 0, 'no'],
                [1, 0, 0, 1, 'no'],
                [1, 1, 1, 1, 'yes'],
                [1, 0, 1, 2, 'yes'],
                [1, 0, 1, 2, 'yes'],
                [2, 0, 1, 2, 'yes'],
                [2, 0, 1, 1, 'yes'],
                [2, 1, 0, 1, 'yes'],
                [2, 1, 0, 2, 'yes'],
                [2, 0, 0, 0, 'no']]
    labels = ['不放贷', '放贷']       #分类属性
    return dataSet, labels           #返回数据集和分类属性
 
"""
函数说明:计算给定数据集的经验熵(香农熵)
Parameters:
    dataSet - 数据集
Returns:
    shannonEnt - 经验熵(香农熵)
"""
def calcShannonEnt(dataSet):
    numEntires = len(dataSet)                 # 返回数据集的行数
    labelCounts = {}                          # 保存每个标签(Label)出现次数的字典
    for featVec in dataSet:                   # 对每组  [0, 0, 0, 0, 'no']  特征向量进行统计
        currentLabel = featVec[-1]            # 提取标签(Label)信息，即得到 yes 和no ，然后在进行统计
        if currentLabel not in labelCounts.keys():    # 如果标签(Label)没有放入统计次数的字典,添加进去
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1         # Label计数
    shannonEnt = 0.0                           # 经验熵(香农熵)
    for key in labelCounts:                    # 计算香农熵
        prob = float(labelCounts[key]) / numEntires    # 选择该标签(Label)的概率
        # 利用香农熵公式，计算经验熵 shannonEnt = - prob*log(prob,2)
        shannonEnt -= prob * log(prob, 2)            # 利用公式计算
    return shannonEnt              # 返回经验熵(香农熵)
 
if __name__ == '__main__':
    # 用两个参数去接受 数集 和 标签
    dataSet, features = createDataSet()
    print(dataSet)
    # 调用calcShannonEnt函数去计算香农熵
    print(calcShannonEnt(dataSet))