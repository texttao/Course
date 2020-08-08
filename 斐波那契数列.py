def fibonList(num):#定义斐波那契数列
    fibonList = []
    for i in range(0, num):
        if len(fibonList) < 2:
            fibonList.append(1)
            continue
        fibonList.append(fibonList[-1]+fibonList[-2])
    return fibonList

print(fibonList(10))