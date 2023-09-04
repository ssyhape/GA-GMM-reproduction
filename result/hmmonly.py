from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt
import geatpy as ea
import random
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm
#生成时域信号序列
# 设置时域长度
T = 50
# 生成时域序列
t = np.arange(T)
# 生成原始信号
x = np.sin(2*np.pi*t/15)
# 添加高斯白噪声
x[:10] = np.random.normal(0, 0.1, 10)
x[40:] = np.random.normal(0, 0.1, 10)
signal =x
signal = LabelEncoder().fit_transform(signal)
signal = np.atleast_2d(signal)

model = hmm.MultinomialHMM(n_components=2, tol=1e-50, n_iter=500,init_params='')
n_futures = len(set(signal[0]))


def normalize_array(arr):
    total = sum(arr)
    return [x / total for x in arr]

@ea.Problem.single
def evalVars(Vars):  # 定义目标函数（含约束）
    pi_ini = Vars[0:2]
    pi_ini = normalize_array(pi_ini)
    B_ini = Vars[6:6 + 2 * n_futures].reshape(2, n_futures)
    B_ini[0] = normalize_array(B_ini[0])
    B_ini[1] = normalize_array(B_ini[1])
    A_ini = Vars[2:6].reshape(2,2)
    A_ini[0] = normalize_array(A_ini[0])
    A_ini[1] = normalize_array(A_ini[1])
    model.startprob_ = pi_ini
    model.transmat_ = A_ini
    model.emissionprob_ = B_ini
    f = model.score(signal.T)#ln(P)
    cv1 = 0.5-A_ini[0][0]
    cv2 = 0.5-A_ini[1][1]
    CV = np.hstack([cv1,cv2])
    return f, CV

problem = ea.Problem(name='soea quick start demo',
                        M=1,  # 目标维数
                        maxormins=[-1],  # 目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标
                        Dim=6+2*n_futures,  # 决策变量维数
                        varTypes=[0 for i in range(6+2*n_futures)],  # 决策变量的类型列表，0：实数；1：整数
                        lb=[0 for i in range(6+2*n_futures)],  # 决策变量下界
                        ub=[1 for i in range(6+2*n_futures)],  # 决策变量上界
                        evalVars=evalVars)
# 构建算法
algorithm = ea.soea_SEGA_templet(problem,
                                    ea.Population(Encoding='RI', NIND=200),
                                    MAXGEN=500,  # 最大进化代数。
                                    logTras=1,  # 表示每隔多少代记录一次日志信息，0表示不记录。
                                    trappedValue=1e-10,  # 单目标优化陷入停滞的判断阈值。
                                    maxTrappedCount=10)  # 进化停滞计数器最大上限值。
# 求解
res = ea.optimize(algorithm, seed=1, verbose=True, drawing=1, outputMsg=True, drawLog=False, saveFlag=True, dirName='result')

best_ans_GA = list(res.items())[4][1][0]
pi_ini = best_ans_GA[0:2]
pi_ini = normalize_array(pi_ini)
B_ini = best_ans_GA[6:6 + 2 * n_futures].reshape(2, n_futures)
B_ini[0] = normalize_array(B_ini[0])
B_ini[1] = normalize_array(B_ini[1])
A_ini = best_ans_GA[2:6].reshape(2, 2)
A_ini[0] = normalize_array(A_ini[0])
A_ini[1] = normalize_array(A_ini[1])
model.startprob_ = pi_ini
model.transmat_ = A_ini
model.emissionprob_ = B_ini
model.fit(signal.T)
ans_loc =model.decode(signal.T)[1]

plt.plot(t,x)
for i in range(len(ans_loc)):
    if ans_loc[i] == 1:
        plt.scatter(i, x[i], color='red')
    else :
        plt.scatter(i,x[i],color='blue')
plt.show()



