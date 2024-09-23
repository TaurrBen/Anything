import time

import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 各型号车辆信息表
table1 = {"型号": [0, 1, 2, 3, 4, 5],  #
          "最大承重": [5, 12, 60, 60, 26, 26],  # t
          "速度": [10, 8.3, 5.6, 4.6, 2.8, 2.3],  # m/s
          "Ca": [19, 2, 8, 2, 8, 1],  # 辆
          "opy": [0, 0, 1, 1, 2, 2],  # 胶轮车0，地轨车1，单轨吊2
          }
# 运单信息表
table2 = {"货物名": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
          "批量": [1, 2, 1, 1, 1,
                 1, 1, 1, 1, 1,
                 1, 7, 1, 1, 12,
                 # 1,7,300,1,12,
                 1],
          "单个重量": [4.26465, 5.6862, 5.0544, 0.1, 0.075,
                   2.6, 0.003159, 14, 12, 4.7,
                   4.8, 0.005, 0.002633 * 300, 11.04, 0.05,
                   # 4.8,0.005,0.002633,11.04,0.05,
                   1],
          "运输地点": ["五度换装站", "五度换装站", "五度换装站", "一采四中", "十一采水仓",
                   "十一采水仓", "五度换装站", "地面料场", "地面料场", "8302上巷",
                   "地面料场", "地面料场", "地面料场", "地面料场", "地面料场",
                   "地面料场"],
          "中转地点": [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],#1中转 0不中转
          "运输地点": ["边界进运二联巷", "边界五中", "边界五中", "八采二中皮带机通道", "地面机修厂",
                   "地面机修厂", "1304上巷", "2304上巷S弯以里", "2304上巷", "沉淀池南通道",
                   "6305制浆车间", "1304老泵站", "边界二中", "边界二中", "边界二中",
                   "6305下巷"],
          "时间限制": [72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72],
          "支分系数": [1.5, 1, 1, 1, 1.5,
                   1.5, 2, 1, 1, 1,
                   1, 1, 1, 1, 1,
                   1],
          "单价": [22, 22, 10, 10, 10,
                 28, 22, 60, 60, 50,
                 50, 4, 0.1, 30, 2.5,
                 30],
          "车1": [2, 2, 2, 2, 2,
                 2, 2, 1, 1, 0,
                 0, 0, 0, 0, 0,
                 0],# 胶轮车0，地轨车1，单轨吊2
          "距离1": [2192, 2238, 2238, 4368, 2487,
                  2487, 2698, 1550, 1550, 1690,
                  5988, 3999, 2669, 2669, 2669,
                  6912],
          "车2": [-1, -1, -1, -1, 1,
                 1, -1, 2, 2, -1,
                 -1, -1, -1, -1, -1,
                 -1],# 胶轮车0，地轨车1，单轨吊2  无中转-1
          "距离2": [0, 0, 0, 0, 1089,
                  1089, 0, 2000, 2000, 0,
                  0, 0, 0, 0, 0,
                  0]
          }
# 故障信息表
table3 = {"故障发生概率": [13 / 30, 1 / 30, 4 / 15],
          "平均影响时间": [63 / 60, 47.8 / 60, 40 / 60],
          }

pdTable1 = pd.DataFrame(table1)
pdTable2 = pd.DataFrame(table2)
pdTable3 = pd.DataFrame(table3)

# 单位中转成本
def e(m, n):
    return [[0, 0, 11.97], [0, 0, 1.87], [11.97, 2.38, 0]][m][n]


# 单位中转时间
def A(m, n):
    return [[0, 0, 0.5], [0, 0, 0.5], [0.5, 0.5, 0]][m][n]


# 转运成本计算
def C1(Table2=pdTable2):
    table = Table2.loc[Table2["中转地点"] == 1]
    m = list(table["车1"])
    n = list(table["车2"])
    P = list(table["批量"])
    W = list(table["单个重量"])
    c1 = 0
    i = 0
    for _ in m:
        c1 += e(m[i], n[i]) * P[i] * W[i]
        i += 1
    return c1


# 运输成本
def C2(Table2=pdTable2):
    qx = list(Table2["单价"])
    P = list(Table2["批量"])
    l = list(Table2["支分系数"])
    c2 = 0
    i = 0
    for _ in qx:
        c2 += qx[i] * P[i] * l[i]
        i += 1
    return c2


# 运输路程时间
def T0(X, Table1=pdTable1, Table2=pdTable2):
    t0_list_x = []
    t0_list = []
    Vf = []
    dis1 = list(Table2["距离1"])
    Ca = np.cumsum(list(Table1["Ca"]))
    P1 = np.cumsum(list(Table2["批量"]))
    P2 = np.cumsum(np.multiply(list(Table2["中转地点"]), list(Table2["批量"])))
    dis2 = list(Table2["距离2"])
    set1 = set()
    i = 0
    for x in X:
        Vf.append(Table1.iloc[np.sum(Ca <= x), 2])  # 得到车速
        if i < P1[-1]:
            index = np.sum(P1 <= i)
            t0 = dis1[index] / Vf[-1] / 3600
        else:
            index = np.sum(P2 <= i - P1[-1])
            t0 = dis2[index] / Vf[-1] / 3600
        # 将同一辆车累加
        if x not in set1:
            t0_list.append(t0)
        t0_list_x.append(t0)
        set1.add(x)
        i += 1
    return t0_list


# 物料转运时间  A为中转用时，默认0.5
def T1(X,Table2=pdTable2, A=0.5):
    t1_list = []
    P1 = np.cumsum(list(Table2["批量"]))
    i = 0
    set1 = set()
    for x in X:
        if i < P1[-1]:
            index = np.sum(P1 <= i)
            t1 = Table2.iloc[index, 4] * A
        else:
            t1 = 0
        if x not in set1:
            t1_list.append(t1)
        set1.add(x)
        i += 1

    return t1_list

# 迎接集中停运时间
def T2(X, C=3):
    t2 = []
    set1 = set()
    for x in X:
        # 某辆车停运
        if x not in set1:
            t2.append(np.random.binomial(1, 1 / 9) * C)
        set1.add(x)
    return t2

# 载具故障维修所需时间
def T3(X, Table1=pdTable1, Table3=pdTable3):
    t3 = []
    Ca = np.cumsum(list(Table1["Ca"]))
    set1 = set()
    for x in X:
        row = Table1.iloc[np.sum(Ca <= x), 4]  # 得到车类型
        p, time = Table3.iloc[row,]  # 得到故障概率和时间
        rand = np.random.binomial(1, p)
        if x not in set1:
            t3.append(rand * time)
        set1.add(x)
    return t3


# 时间成本
def C3(X):
    t0 = sum(T0(X))
    t1 = sum(T1(X))
    t2 = sum(T2(X))
    t3 = sum(T3(X))
    # print("t0,t1,t2,t3,", t0, t1, t2, t3)
    c3 = (t0 + t1 + t2 + t3)
    return c3


# 优化函数
def Fun(X,Z=38):
    c1 = C1()
    c2 = C2()
    c3 = C3(X)
    # print("c1,c2,c3", c1, c2, c3, len(set(X)))
    return (c1 + c2 + c3*Z) * len(set(X)),c3


# 随机解X  将批量展开，中转后的车展开，得到38维度
def initialX(Table1=pdTable1, Table2=pdTable2):
    Ca = np.cumsum(list(Table1["Ca"]))
    P1 = np.cumsum(list(Table2["批量"]))
    P2 = np.cumsum(np.multiply(list(Table2["中转地点"]), list(Table2["批量"])))
    dim = P1[-1] + P2[-1]
    ub = []# 上边界
    lb = []# 下边界
    X = []
    set1 = set()
    for i in range(0, dim):
        if i < P1[-1]:
            index = np.sum(P1 <= i)
            type = Table2.iloc[index, 8]
        else:
            index = np.sum(P2 <= i - P1[-1])
            type = Table2.iloc[index, 10]
        row = list(Table1.loc[Table1["opy"] == type].index)
        ub.append(Ca[row[1]])
        if row[0] == 0:
            lb.append(0)
        else:
            lb.append(Ca[row[0] - 1])

        if i < P1[-1]:
            # 距离相同？
            if Table2.iloc[index, 9] not in set1:
                x = np.random.randint(lb[-1], ub[-1])
                while x in X:
                    x = np.random.randint(lb[-1], ub[-1])
            else:
                if np.random.binomial(1, 3 / 4) == 1:
                    x = x
                else:
                    while x in X:
                        x = np.random.randint(lb[-1], ub[-1])
            set1.add(Table2.iloc[index, 9])
        else:
            if Table2.iloc[index, 11] not in set1:
                x = np.random.randint(lb[-1], ub[-1])
                while x in X:
                    x = np.random.randint(lb[-1], ub[-1])
            else:
                if np.random.binomial(1, 3 / 4) == 1:
                    x = x
                else:
                    t = 0
                    while x in X:
                        t += 1
                        if t > 30:
                            break
                        x = np.random.randint(lb[-1], ub[-1])
            set1.add(Table2.iloc[index, 11])

        X.append(x)
    return X,lb,ub


# 约束条件
def constraints(X, Table1=pdTable1, Table2=pdTable2):
    # 限重
    Ca = np.cumsum(list(Table1["Ca"]))
    P1 = np.cumsum(list(Table2["批量"]))
    P2 = np.cumsum(np.multiply(list(Table2["中转地点"]), list(Table2["批量"])))
    for x in X:
        weight = 0
        i = 0
        for x1 in X:
            if x1 == x:
                if i < P1[-1]:
                    index = np.sum(P1 <= i)
                else:
                    index = np.sum(P2 <= i - P1[-1])
                weight += Table2.iloc[index, 2]
            i += 1
        Wmax = Table1.iloc[np.sum(Ca <= x), 1]  # 得到车类型
        if weight > Wmax:
            return False

    # 限时
    t0 = T0(X)
    t1 = T1(X)
    t2 = T2(X)
    t3 = T3(X)
    time = t0 + t1 + t2 + t3
    x_table =[]
    set1 = set()
    for x in X:
        if x not in set1:
            x_table.append(x)
        set1.add(x)

    j = 0
    for x in x_table:
        i = X.index(x)
        if i < P1[-1]:
            index = np.sum(P1 <= i)
        else:
            index = np.sum(P2 <= i - P1[-1])
        Tmax = Table2.iloc[index, 6]
        if time[j] > Tmax:
            return False
        j += 1

    return True

#
# X = initialX()
# while constraints(X) == False:
#     X = initialX()
# X = [31,32,32,32,33,  # 31运订单1    32，32运订单2（两个批量）   32运订单3   33运订单4
#      34,34,35,21,21,  # 34运订单5    34运订单6     35运订单7   21运订单8    21运订单9
#      0,1,2,2,2,
#      2,2,2,2,3,
#      19,3,3,3,3,
#      3,3,3,3,3,
#      3,3,3,4,22,
#      22,36,36]

# 例X 中 前34个为运输第一个地点，后4个为运输中转后第二个地点

# print(Fun(X))


'''优化函数'''
# #,y = x^2     用户可以自己定义其他函数
# Y1=[10,3,15,2]
# def fun(X):
#     output = (sum(np.square(X-Y1))/len(X))**(1/2)
#     return output


''' 种群初始化函数 '''


def initial(pop, dim):
    X = [[0]*dim]*pop
    for i in range(pop):
        X[i],lb,ub=initialX()
        while constraints(X[i]) == False:
            X[i],lb,ub = initialX()

    return X, lb, ub


'''边界检查函数'''


def BorderCheck(X, Ub, Lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            ub = Ub[j]
            lb = Lb[j]
            x = X[i][j]
            if (X[i][j] >= ub) or (X[i][j] < lb):
                X[i], Ub, Lb = initialX()
                break
        while constraints(X[i]) == False:
            X[i], Ub, Lb = initialX()
    return X


'''计算适应度函数'''


def CaculateFitness(pop,X, fun):
    fitness = [0]*pop
    c3 = [0]*pop
    for i in range(pop):
        fitness[i],c3[i] = fun(X[i])
    return fitness,c3


'''适应度排序'''


def SortFitness(Fit):
    fitness = list(np.sort(Fit, axis=0))
    index = list(np.argsort(Fit, axis=0))
    return fitness, index


'''根据适应度对位置进行排序'''
def SortPosition(pop, dim,X,c3, index):
    Xnew = [[0]*dim]*pop
    c3new = [0]*pop
    for i in range(pop):
        Xnew[i] = X[index[i]]
        c3new[i] = c3[index[i]]
    return Xnew,c3new


'''鲸鱼优化算法'''
def WOA(pop, dim,MaxIter, fun):
    X, lb, ub = initial(pop, dim)  # 初始化种群
    fitness,c3 = CaculateFitness(pop,X, fun)  # 计算适应度值
    fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
    X,c3 = SortPosition(pop, dim,X,c3, sortIndex)  # 种群排序
    GbestScore = fitness[0]
    GbestPositon = np.zeros([1, dim])
    GbestPositon[0, :] = X[0]
    Curve = np.zeros([MaxIter, 1])
    c3_total = 0
    for t in range(MaxIter):
        print(t," out of ",MaxIter)
        c3_total += sum(c3)/pop
        Leader = X[0]  # 领头鲸鱼
        a = 2 - t * (2 / MaxIter)  # 线性下降权重2 - 0
        a2 = -1 + t * (-1 / MaxIter)  # 线性下降权重-1 - -2
        for i in range(pop):
            r1 = random.random()
            r2 = random.random()

            A = 2 * a * r1 - a
            C = 2 * r2
            b = 1
            l = (a2 - 1) * random.random() + 1

            for j in range(dim):

                p = random.random()
                if p < 0.5:
                    if np.abs(A) >= 1:
                        rand_leader_index = min(int(np.floor(pop * random.random() + 1)), pop - 1)
                        X_rand = X[rand_leader_index]
                        D_X_rand = np.abs(C * X_rand[j] - X[i][j])
                        X[i][j] = int(X_rand[j] - A * D_X_rand)
                    elif np.abs(A) < 1:
                        D_Leader = np.abs(C * Leader[j] - X[i][j])
                        X[i][j] = int(Leader[j] - A * D_Leader)
                elif p >= 0.5:
                    distance2Leader = np.abs(Leader[j] - X[i][j])
                    X[i][j] = int(distance2Leader * np.exp(b * l) * np.cos(l * 2 * math.pi) + Leader[j])

        X = BorderCheck(X, ub, lb, pop, dim)  # 边界检测
        fitness, c3 = CaculateFitness(pop, X, fun)  # 计算适应度值
        fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
        X, c3 = SortPosition(pop, dim, X, c3, sortIndex)
        if fitness[0] <= GbestScore:  # 更新全局最优
            GbestScore = fitness[0]
            GbestPositon[0, :] = X[0]
        Curve[t] = GbestScore
    print("平均配送时间", c3_total / MaxIter, "小时")
    return GbestScore, GbestPositon, Curve


'''主函数 '''
# 设置参数
starttime = time.time()
pop = 10  # 种群数量
MaxIter = 10  # 最大迭代次数
P1 = np.cumsum(list(pdTable2["批量"]))
P2 = np.cumsum(np.multiply(list(pdTable2["中转地点"]), list(pdTable2["批量"])))
dim = P1[-1] + P2[-1]
# dim = 4  # 维度

GbestScore, GbestPositon, Curve = WOA(pop, dim, MaxIter, Fun)
time = (time.time()-starttime)/60
print('程序耗费时间:',time,"分钟")
print('最优适应度值：', GbestScore)
print('最优解：', GbestPositon)

# 绘制适应度曲线
plt.figure(1)
plt.plot(Curve, 'r-', linewidth=2)
plt.xlabel('Iteration', fontsize='medium')
plt.ylabel("Fitness", fontsize='medium')
plt.grid()
plt.title('WOA', fontsize='large')
plt.show()
# 绘制搜索空间

# X = np.arange(-4, 4, 0.25)
# Y = np.arange(-4, 4, 0.25)
# X, Y = np.meshgrid(X, Y)
# Z = X ** 2 + Y ** 2
#
# fig = plt.figure(2)
# ax = fig.add_subplot(111, projection='3d')#这句话原文是错的，！我学的改正确了！！
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
# plt.show()
