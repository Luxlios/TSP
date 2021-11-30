# -*- coding = utf-8 -*-
# @Time : 2021/11/21 12:09
# @Author : Luxlios
# @File : particle_swarm_a.py
# @Software : PyCharm

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

# 获取交换子序列的函数
def get_swap(x, best):
    # x为某个序列，best为另一个序列（pbest，gbest）
    # 输出x到best的交换子序列
    swap = []
    for i in range(len(x)):
        if x[i] != best[i]:
            j = x.index(x == best[i])
            swap_temp = [i, j].copy()
            swap.append(swap_temp)
            x[i], x[j] = x[j], x[i]
    return swap


# 进行交换子序列的函数
def do_swap(x, swap, c):
    # x为某个序列
    # swap为交换子序列
    # c为进行交换子序列的概率
    for i, j in swap:
        rand = random.random()
        if rand <= c:
            x[i], x[j] = x[j], x[i]
    return x


# 粒子群算法TSP
def particle_swarm_a(data, num_particle, c1, c2, iteration):
    # 输入data为城市经纬度
    # num_particle为粒子的数目
    # c1为认知执行概率
    # c2为社会执行概率
    # 返回最优距离，最优方案和历代平均适应度

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

    # 生成初始粒子群
    initial = []
    initial_temp = list(range(num_city))
    for i in range(num_particle):
        random.shuffle(initial_temp)
        while initial_temp in initial:
            random.shuffle(initial_temp)
        initial_temp1 = initial_temp.copy()
        initial.append(initial_temp1)

    swarm = initial.copy()
    # 适应度计算，适应度为距离
    adaption = []
    for i in range(num_particle):
        d = 0
        for j in range(num_city - 1):
            d = d + distance[swarm[i][j]][swarm[i][j + 1]]
        d = d + distance[swarm[i][num_city - 1]][swarm[i][0]]  # 返回第一个地方
        adaption_temp = d.copy()
        adaption.append(adaption_temp)

    # 初始化粒子本身经历的最优位置pbest和群体目前经历的最有位置gbest
    pbest_adaption = adaption
    pbest = swarm

    gbest_adaption = min(adaption)
    gbest = swarm[adaption.index(min(adaption))]

    flag = True
    itera = 1
    adaption_history = []
    # 主体循环
    while flag:
        # 保存历代平均适应度
        adaption11 = np.mean(adaption).copy()
        adaption_history.append(adaption11)

        # 更新粒子的位置
        for i in range(num_particle):
            x1 = swarm[i].copy()
            swap1 = get_swap(x1, pbest[i])
            swap2 = get_swap(x1, gbest)
            swarm[i] = do_swap(swarm[i], swap1, c1)
            swarm[i] = do_swap(swarm[i], swap2, c2)

            # 计算粒子更新得到的适应度
            d = 0
            for j in range(num_city - 1):
                d = d + distance[swarm[i][j]][swarm[i][j + 1]]
            d = d + distance[swarm[i][num_city - 1]][swarm[i][0]]  # 返回第一个地方
            adaption1 = d.copy()
            # 更新粒子pbest
            if adaption1 < pbest_adaption[i]:
                pbest_adaption[i] = adaption1
                pbest[i] = swarm[i]
        # 更新种群gbest
        gbest_adaption1 = min(pbest_adaption)
        if gbest_adaption1 < gbest_adaption:
            gbest_adaption = gbest_adaption1
            gbest = pbest[pbest_adaption.index(min(pbest_adaption))]

        # 适应度计算，适应度为距离
        adaption = []
        for i in range(num_particle):
            d = 0
            for j in range(num_city - 1):
                d = d + distance[swarm[i][j]][swarm[i][j + 1]]
            d = d + distance[swarm[i][num_city - 1]][swarm[i][0]]  # 返回第一个地方
            adaption_temp = d.copy()
            adaption.append(adaption_temp)

        if itera >= iteration:
            print('最小距离:', gbest_adaption)
            print('方案:', gbest)
            flag = False
        else:
            itera += 1

    return gbest_adaption, gbest, adaption_history

if __name__ == '__main__':
    data = pd.read_excel('data.xlsx')
    data = np.array(data)
    data = data[:, [1, 2]]
    [x3, y3, adaption3] = particle_swarm_a(data, 500, 0.4, 0.6, 300)
    plt.figure()
    plt.plot(range(len(adaption3)), adaption3, c='C1', lw=1, ls='-')
    plt.title('particle_swarm_a_adaption curve')
    plt.xlabel('iteration')
    plt.ylabel('average adaption')
    plt.show()