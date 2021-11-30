# -*- coding = utf-8 -*-
# @Time : 2021/11/21 12:02
# @Author : Luxlios
# @File : genetic_a.py
# @Software : PyCharm

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

# 遗传算法TSP
def genetic_a(data, population, cross_rate, mutate_rate, iteration):
    # 输入data为各地点的经纬度
    # population为遗传算法种群大小
    # cross_rate为交叉的概率
    # mutate_rate为变异的概率
    # iteration为最大迭代次数
    # 返回最优距离，最优方案和历代平均距离

    # 计算城市距离矩阵
    num_city = data.shape[0]
    distance = []
    for i in range(num_city):
        distance_temp = []
        for j in range(num_city):
            d = np.sqrt((data[i][0]-data[j][0])**2 + (data[i][1]-data[j][1])**2)
            d1 = d.copy()
            distance_temp.append(d1)
        distance_temp1 = distance_temp.copy()
        distance.append(distance_temp1)

#     print(distance)
    # 生成初始种群
    initial = []
    initial_temp = list(range(num_city))
    for i in range(population):
        random.shuffle(initial_temp)
        while(initial_temp in initial):
            random.shuffle(initial_temp)
        initial_temp1 = initial_temp.copy()
        initial.append(initial_temp1)

    popul = initial.copy()
    # print(popul)
    # 适应度计算
    # 适应度为总距离的倒数
    adaption = []
    for i in range(population):
        d = 0
        for j in range(num_city-1):
            d = d + distance[popul[i][j]][popul[i][j+1]]
        d = d + distance[popul[i][num_city-1]][popul[i][0]]   # 返回第一个地方
        adaption_temp = 100000.0/d
        adaption_temp1 = adaption_temp.copy()
        adaption.append(adaption_temp1)

#     print(adaption)
    flag = True
    itera = 0
    adaption_history = []
    # 主体循环
    while flag:
        # 保存历代平均适应度
        adaption11 = np.mean(adaption).copy()
        adaption_history.append(adaption11)

        # 更新种群
        # 轮盘赌法选择个体，得到与原种群数目相等的新种群
        popul_temp = []
        gamble = []
        gamble_temp = 0
        for i in range(population):
            gamble_temp = gamble_temp + adaption[i]
            gamble_temp1 = gamble_temp.copy()
            gamble.append(gamble_temp1)
        for i in range(population):
            rand = random.uniform(0, gamble[population-1])   # 产生0-gamble[population-1]的随机数
            for j in range(population):
                if rand < gamble[j]:       # 寻找个体
                    popul1 = popul[j].copy()
                    popul_temp.append(popul1)
                    break
                else:
                    continue
        # TSP问题交叉和变异一般是个体自己与自己作用
        # 交叉，做五次交换
        for i in range(population):
            rand = random.random()
            if rand < cross_rate:
                for k in range(5):
                    m = random.randint(0, num_city-1)
                    n = random.randint(0, num_city-1)
                    popul_temp[i][m], popul_temp[i][n] = popul_temp[i][n], popul_temp[i][m]
        # 变异
        for i in range(population):
            rand = random.random()
            if rand < mutate_rate:
                m = random.randint(0, num_city - 1)
                n = random.randint(0, num_city - 1)
                popul_temp[i][m], popul_temp[i][n] = popul_temp[i][n], popul_temp[i][m]

        # 适应度计算
        # 适应度为总距离的倒数
        adaption1 = []
        for i in range(population):
            d = 0
            for j in range(num_city-1):
                d = d + distance[popul_temp[i][j]][popul_temp[i][j+1]]
            d = d + distance[popul_temp[i][num_city-1]][popul_temp[i][0]]   # 返回第一个地方
            adaption1_temp = 100000.0/d
            adaption1_temp1 = adaption1_temp.copy()
            adaption1.append(adaption1_temp1)
        if (np.linalg.norm(np.array(adaption) - np.array(adaption1)) <= 0.001) or (itera >= iteration):
            print('最小距离:', 100000.0/max(adaption))
            m_index = adaption1.index(max(adaption1))
            print('方案:', popul_temp[m_index])
            flag = False
        else:
            popul = popul_temp
            adaption = adaption1
            itera += 1
#         print(popul)
    return 100000.0/max(adaption), popul_temp[m_index], adaption_history

if __name__ == '__main__':
    data = pd.read_excel('data.xlsx')
    data = np.array(data)
    data = data[:, [1, 2]]
    [x1, y1, adaption1] = genetic_a(data, 500, 0.1, 0.01, 5000)
    plt.figure()
    plt.plot(range(len(adaption1)), adaption1, c='C1', lw=1, ls='-')
    plt.title('genetic_a_adaption curve')
    plt.xlabel('iteration')
    plt.ylabel('fitness')
    plt.show()