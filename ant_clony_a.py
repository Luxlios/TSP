# -*- coding = utf-8 -*-
# @Time : 2021/11/21 12:08
# @Author : Luxlios
# @File : ant_clony_a.py
# @Software : PyCharm

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

# 蚁群算法TSP
def ant_clony_a(data, num_ant, alpha, beta, rou, iteration):
    # 输入data为各地点经纬度
    # num_ant为蚁群算法蚁群大小
    # alpha为信息素重要程度因子
    # beta为启发函数重要因子
    # rou为信息素挥发系数
    # iteration为最大迭代次数
    # 返回最优距离，最优方案和历代平均距离

    # 计算城市距离矩阵
    num_city = data.shape[0]
    distance = []
    for i in range(num_city):
        distance_temp = []
        for j in range(num_city):
            d = np.sqrt((data[i][0] - data[j][0]) ** 2 + (data[i][1] - data[j][1]) ** 2)
            distance_temp.append(d)
        distance_temp1 = distance_temp.copy()
        distance.append(distance_temp1)

    # 计算启发矩阵，表示ant从城市i到城市j的期望程度
    # 注意对角线元素为0，不能直接除
    heuristic = 1.0 / (np.array(distance) + np.diag([1] * num_city))
    heuristic = heuristic - np.diag([1] * num_city)

    # 初始化信息素浓度矩阵,全1
    pheromone = np.ones([num_city, num_city])
    # 初始化ant的路径，从第一个城市（0城市）出发
    path = np.zeros([num_ant, num_city], dtype=int)

    flag = True
    itera = 1
    distance_history = []
    # 主体循环
    while flag:
        # 轮盘赌法选择每只蚂蚁走的路径，从城市0开始
        for i in range(num_ant):
            visited = 0
            unvisited = list(range(1, num_city))
            for j in range(1, num_city):
                gamble = []
                gamble_temp = 0
                for k in range(len(unvisited)):
                    gamble_temp = gamble_temp + (pheromone[visited][unvisited[k]] ** alpha) * (
                                heuristic[visited][unvisited[k]] ** beta)
                    gamble_temp1 = gamble_temp.copy()
                    gamble.append(gamble_temp1)

                rand = random.uniform(0, gamble[len(unvisited) - 1])  # 产生0-gamble[len(unvisited)-1]的随机数
                for k in range(len(gamble)):
                    if rand < gamble[k]:  # 寻找个体
                        visit_next = unvisited[k]
                        break
                    else:
                        continue
                path[i, j] = visit_next
                unvisited.remove(visit_next)
                visited = visit_next

        # 计算每只蚂蚁总距离矩阵
        ant_distance = []
        for i in range(num_ant):
            d = 0
            for j in range(num_city - 1):
                d = d + distance[path[i][j]][path[i][j + 1]]
            d = d + distance[path[i][num_city - 1]][path[i][0]]  # 返回第一个地方
            d1 = d.copy()
            ant_distance.append(d1)

        # 保存历代平均距离
        ant_distance1 = np.mean(ant_distance).copy()
        distance_history.append(ant_distance1)

        # 更新信息素矩阵
        # 采用Ant-Quantity模型模拟释放信息素的浓度
        Q = 1
        pheromone1 = np.zeros([num_city, num_city])
        for i in range(num_ant):
            for j in range(num_city - 1):
                pheromone1[path[i][j]][path[i][j + 1]] += Q / distance[path[i][j]][path[i][j + 1]]
            pheromone1[path[i][num_city - 1]][path[i][0]] += Q / distance[path[i][num_city - 1]][path[i][0]]
        pheromone = (1 - rou) * pheromone + pheromone1

        if itera >= iteration:
            print('最小距离:', min(ant_distance))
            m_index = ant_distance.index(min(ant_distance))
            print('方案:', list(path[m_index]))
            flag = False
        else:
            itera += 1

    return min(ant_distance), path[m_index], distance_history

if __name__ == '__main__':
    data = pd.read_excel('data.xlsx')
    data = np.array(data)
    data = data[:, [1, 2]]
    [x2, y2, distance2] = ant_clony_a(data, 500, 1, 1, 0.2, 100)
    plt.figure()
    plt.plot(range(len(distance2)), distance2, c='C1', lw=1, ls='-')
    plt.title('ant_clony_a_distance curve')
    plt.xlabel('iteration')
    plt.ylabel('average distance')
    plt.show()
