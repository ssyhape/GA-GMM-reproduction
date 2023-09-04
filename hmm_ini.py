import numpy as np

def normalize_array(arr):
    total = sum(arr)
    return [x / total for x in arr]
def forward(obs, states, start_prob, trans_prob, emit_prob):
    """
    obs:观测序列
    states：观测序列
    start_prob:初始状态值
    trans_prob:转移状态序列
    emit_prob:观测状态矩阵
    """
    alpha = np.zeros((len(obs), len(states)))
    alpha[0] = start_prob * emit_prob[:, obs[0]]
    for t in range(1,len(obs)):
        for j in range(len(states)):
            alpha[t, j] = emit_prob[j, obs[t]] * np.sum(alpha[t-1,:] * trans_prob[:, j])
    return alpha

def backward(obs, states, trans_prob, emit_prob):
    beta = np.zeros((len(obs), len(states)))
    beta[-1,:] = 1
    for t in range(len(obs)-2, -1, -1):
        for j in range(len(states)):
            beta[t, j] = np.sum(beta[t+1,:] * emit_prob[:, obs[t+1]] * trans_prob[j, :])
    return beta
def viterbi(obs, states, start_prob, trans_prob, emit_prob):
    """
    维特比算法实现
    :param obs: 观测序列，类型为列表
    :param states: 隐状态，类型为numpy数组
    :param start_prob: 初始状态概率向量，类型为numpy数组
    :param trans_prob: 状态转移概率矩阵，类型为numpy二维数组
    :param emit_prob: 发射概率矩阵，类型为numpy二维数组
    :return: 返回最优路径，类型为列表
    """
    # 初始化
    V = [{}] # 存储所有状态的最大概率
    path = {} # 存储每个状态的最优前一个状态
    for y in states:
        V[0][y] = start_prob[y] * emit_prob[y][obs[0]]
        path[y] = [y]
    # 递推
    for t in range(1,len(obs)):
        V.append({})
        new_path = {}
        for y in states:
            # 获取上一个状态的最优状态和概率
            (prob, state) = max([(V[t-1][y0] * trans_prob[y0][y] * emit_prob[y][obs[t]], y0) for y0 in states])
            V[t][y] = prob
            new_path[y] = path[state] + [y]
        path = new_path
    # 终止
    (prob, state) = max([(V[len(obs)-1][y], y) for y in states])
    return path[state]


def baum_welch(obs, states, start_prob, trans_prob, emit_prob, max_iter=100):
    for n in range(max_iter):
        alpha = forward(obs, states, start_prob, trans_prob, emit_prob)
        beta = backward(obs, states, trans_prob, emit_prob)
        xi = np.zeros((len(obs)-1, len(states), len(states)))
        for t in range(len(obs)-1):
            denom = np.sum(np.outer(alpha[t], beta[t+1]) * trans_prob * emit_prob[:, obs[t+1]])
            for i in range(len(states)):
                numer = alpha[t, i] * beta[t+1] * emit_prob[i, obs[t+1]] * trans_prob[i, :]
                xi[t, i, :] = numer / denom
        gamma = np.zeros((len(obs),len(states)))
        #gamma矩阵计算
        for t in range(len(obs)):
            for i in range(len(states)):
                numerator = alpha[t][i] * beta[t][i]
                denominator = 0.
                for j in range(len(V)):
                    denominator += (alpha[t][j] * beta[t][j])
                gamma[t][i] = numerator / denominator#
        # 更新参数
        start_prob = gamma[0, :]
        trans_prob = np.sum(xi, axis=0) / np.sum(gamma, axis=0).reshape((-1, 1))
        for i in range(len(states)):
            trans_prob[i] = normalize_array(trans_prob[i])
        emit_prob = np.copy(emit_prob)
        #emit_prob更新
        for j in range(len(states)):
            for k in range(emit_prob.shape[1]):
                numerator = 0.
                denominator = 0.
                for t in range(len(obs)):
                    if obs[t] == V[k]:
                        numerator += gamma[t, j]
                    denominator += gamma[t, j]
                emit_prob[j][k] = numerator / denominator

    return start_prob, trans_prob, emit_prob


obs = [0, 1, 0, 2]
V = [0,1,2]
states = np.array([0, 1, 2])
start_prob = np.array([0.5, 0.2, 0.3])
trans_prob = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.3, 0.3, 0.4]])
emit_prob = np.array([[0.5, 0.2,0.3], [0.4, 0.4,0.2], [0.3, 0.3,0.4]])
start_prob_new, trans_prob_new, emit_prob_new = baum_welch(obs, states, start_prob, trans_prob, emit_prob)
print('start_prob_new:', start_prob_new)
print('trans_prob_new:', trans_prob_new)
print('emit_prob_new:', emit_prob_new)

state_hat = viterbi(obs,states,start_prob_new,trans_prob_new,emit_prob_new)
print("state_hat",state_hat)



