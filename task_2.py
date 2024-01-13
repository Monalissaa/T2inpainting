# 引入必要的库
import math

# 输入num和str_input
num = int(input())
str_input = input()

# 初始化列表list_a和计数器counter
list_a = []
counter = 0

# 遍历字符串str_input
for index in range(num):
    if str_input[index] == '1':
        counter += 1  # 如果字符是'1'，计数器加1
    elif counter:
        list_a.append(counter)  # 如果字符不是'1'且计数器不为0，将计数器的值添加到列表list_a中，并重置计数器
        counter = 0

# 如果计数器不为0，将其值添加到列表list_a中
if counter:
    list_a.append(counter)

# 如果列表list_a的长度小于等于1，输出列表list_a的长度
if len(list_a) <= 1:
    print(len(list_a))
else:
    # 初始化min_days
    min_days = 100000000

    # 计算每一个全1区间对应着发病最多可以是多少天
    for idx in range(len(list_a)):
        # 首尾特判
        if idx == 0 and str_input[0] == '1':
            now_days = list_a[idx] - 1
        elif idx == len(list_a) - 1 and str_input[-1] == '1':
            now_days = list_a[idx] - 1
        else:
            now_days = (list_a[idx] - 1) // 2
        min_days = min(min_days, now_days)

    # 初始化result
    result = 0

    # 现在度过了min_days天，计算初始的患病牛
    for idx in range(len(list_a)):
        # 长度为list_a[idx]的1，经过min_days天构成，求最少患病牛数量
        result += list_a[idx] // (2 * min_days + 1)
        # 向上取整
        if list_a[idx] % (2 * min_days + 1):
            result += 1

    # 输出结果
    print(result)
