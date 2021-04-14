import numpy as np

'''
   func: 计算二分类结果-混淆矩阵的四个元素
   inputs:
       obs: 观测值，即真实值；
       pre: 预测值；
       threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。

   returns:
       hits, misses, falsealarms, correctnegatives
       #aliases: TP, FN, FP, TN
   '''
def prep_clf(obs, pre, threshold=0.1):
    # 根据阈值分类为 0, 1
    obs = np.where(obs >= threshold, 1, 0)
    pre = np.where(pre >= threshold, 1, 0)

    # True positive (TP)
    hits = np.sum((obs == 1) & (pre == 1))

    # False negative (FN)
    misses = np.sum((obs == 1) & (pre == 0))

    # False positive (FP)
    falsealarms = np.sum((obs == 0) & (pre == 1))

    # True negative (TN)
    correctnegatives = np.sum((obs == 0) & (pre == 0))

    return hits, misses, falsealarms, correctnegatives



# func: 计算精确度precision: TP / (TP + FP)
# inputs:
#     obs: 观测值，即真实值；
#     pre: 预测值；
#     threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
#
# returns:
#     dtype: float
def precision(obs, pre, threshold=0.1):
        TP, FN, FP, TN = prep_clf(obs=obs, pre=pre, threshold=threshold)
        return TP / (TP + FP)



# func: 计算召回率recall: TP / (TP + FN)
# inputs:
#     obs: 观测值，即真实值；
#     pre: 预测值；
#     threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
#
# returns:
#     dtype: float
#
def recall(obs, pre, threshold=0.1):
        TP, FN, FP, TN = prep_clf(obs=obs, pre=pre, threshold=threshold)
        return TP / (TP + FN)


#
# func: 计算准确度Accuracy: (TP + TN) / (TP + TN + FP + FN)
# inputs:
#     obs: 观测值，即真实值；
#     pre: 预测值；
#     threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
#
# returns:
#     dtype: float

def ACC(obs, pre, threshold=0.1):
        TP, FN, FP, TN = prep_clf(obs=obs, pre=pre, threshold=threshold)
        return (TP + TN)/ (TP + TN + FP + FN)



# func:计算f1 score = 2 * ((precision * recall) / (precision + recall))
#
def FSC(obs, pre, threshold=0.1):
        precision_socre = precision(obs, pre, threshold=threshold)
        recall_score = recall(obs, pre, threshold=threshold)
        return 2 * ((precision_socre * recall_score) / (precision_socre + recall_score))



# func: 计算TS评分: TS = hits/(hits + falsealarms + misses)
# 	  alias: TP/(TP+FP+FN)
# inputs:
#     obs: 观测值，即真实值；
#     pre: 预测值；
#     threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
# returns:
#     dtype: float
def TS(obs, pre, threshold=0.1):
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre=pre, threshold=threshold)
    return hits / (hits + falsealarms + misses)



# ETS - Equitable Threat Score
# details in the paper:
# Winterrath, T., & Rosenow, W. (2007). A new module for the tracking of
# radar-derived precipitation with model-derived winds.
# Advances in Geosciences,10, 77–83. https://doi.org/10.5194/adgeo-10-77-2007
# Args:
#     obs (numpy.ndarray): observations
#     pre (numpy.ndarray): prediction
#     threshold (float)  : threshold for rainfall values binaryzation
#                          (rain/no rain)
# Returns:
#     float: ETS value
def ETS(obs, pre, threshold=0.1):
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,
                                                           threshold=threshold)
    num = (hits + falsealarms) * (hits + misses)
    den = hits + misses + falsealarms + correctnegatives
    Dr = num / den

    ETS = (hits - Dr) / (hits + misses + falsealarms - Dr)
    return ETS


# func: 计算误警率。falsealarms / (hits + falsealarms)
# FAR - false alarm rate
# Args:
#     obs (numpy.ndarray): observations
#     pre (numpy.ndarray): prediction
#     threshold (float)  : threshold for rainfall values binaryzation
#                          (rain/no rain)
# Returns:
#     float: FAR value
def FAR(obs, pre, threshold=0.1):
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,
                                                           threshold=threshold)
    return falsealarms / (hits + falsealarms)


# func : 计算漏报率 misses / (hits + misses)
# MAR - Missing Alarm Rate
# Args:
#     obs (numpy.ndarray): observations
#     pre (numpy.ndarray): prediction
#     threshold (float)  : threshold for rainfall values binaryzation
#                          (rain/no rain)
# Returns:
#     float: MAR value
def MAR(obs, pre, threshold=0.1):
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,
                                                           threshold=threshold)
    return misses / (hits + misses)



# func : 计算命中率 hits / (hits + misses)
# pod - Probability of Detection
# Args:
#     obs (numpy.ndarray): observations
#     pre (numpy.ndarray): prediction
#     threshold (float)  : threshold for rainfall values binaryzation
#                          (rain/no rain)
# Returns:
#     float: PDO value
def POD(obs, pre, threshold=0.1):
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,
                                                           threshold=threshold)
    return hits / (hits + misses)



# func: 计算Bias评分: Bias =  (hits + falsealarms)/(hits + misses)
# 	  alias: (TP + FP)/(TP + FN)
# inputs:
#     obs: 观测值，即真实值；
#     pre: 预测值；
#     threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
# returns:
#     dtype: float
def BIAS(obs, pre, threshold = 0.1):
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,
                                                           threshold=threshold)
    return (hits + falsealarms) / (hits + misses)


# HSS - Heidke skill score
# Args:
#     obs (numpy.ndarray): observations
#     pre (numpy.ndarray): pre
#     threshold (float)  : threshold for rainfall values binaryzation
#                          (rain/no rain)
# Returns:
#     float: HSS value
def HSS(obs, pre, threshold=0.1):
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,
                                                           threshold=threshold)

    HSS_num = 2 * (hits * correctnegatives - misses * falsealarms)
    HSS_den = (misses**2 + falsealarms**2 + 2*hits*correctnegatives +
               (misses + falsealarms)*(hits + correctnegatives))

    return HSS_num / HSS_den



# BSS - Brier skill score
# Args:
#     obs (numpy.ndarray): observations
#     pre (numpy.ndarray): prediction
#     threshold (float)  : threshold for rainfall values binaryzation
#                          (rain/no rain)
# Returns:
#     float: BSS value
def BSS(obs, pre, threshold=0.1):
    obs = np.where(obs >= threshold, 1, 0)
    pre = np.where(pre >= threshold, 1, 0)

    obs = obs.flatten()
    pre = pre.flatten()

    return np.sqrt(np.mean((obs - pre) ** 2))



# Mean absolute error
# Args:
#     obs (numpy.ndarray): observations
#     pre (numpy.ndarray): prediction
# Returns:
#     float: mean absolute error between observed and simulated values
def MAE(obs, pre):

    obs = pre.flatten()
    pre = pre.flatten()

    return np.mean(np.abs(pre - obs))



# Root mean squared error
# Args:
#     obs (numpy.ndarray): observations
#     pre (numpy.ndarray): prediction
# Returns:
#     float: root mean squared error between observed and simulated values
def RMSE(obs, pre):

    obs = obs.flatten()
    pre = pre.flatten()

    return np.sqrt(np.mean((obs - pre) ** 2))
