import numpy as np
import math

# inpXarr = []
# inpYarr = []

# print("q - выход")
# while input() != "q":  
#   print("Введите элемент класса '0 1'")
#   element = [int(el) for el in input().split()]
#   inpXarr.append(element)
#   print("Введите класс элемента -1 - А, 1 - В")
#   inpYarr.append(int(input()))

# print(inpXarr)
# print(inpYarr)
x_train = np.array([[10, 5], [12, 8], [20, 10], [22, 21], [11, 6], [12, 7], [21, 11], [19, 21]])
y_train = np.array([1, 1, -1, -1, 1, 1, -1, -1])

# мат ожидание
mw1, ml1 = np.mean(x_train[y_train == 1], axis=0)
mw_1, ml_1 = np.mean(x_train[y_train == -1], axis=0)

# дисперсия
sw1, sl1 = np.var(x_train[y_train == 1], axis=0)
sw_1, sl_1 = np.var(x_train[y_train == -1], axis=0)

print('МО: ', mw1, ml1, mw_1, ml_1)
print('Дисперсии:', sw1, sl1, sw_1, sl_1)

# вероятность ошибки
for x in x_train[y_train == 1]:
  print(x)
  BpErrWeight = (1 / (math.sqrt(2*math.pi)*sw1)) * math.exp(-0.5*((x[0] - mw1)/sw1)**2)
  BpErrHeight = (1 / (math.sqrt(2*math.pi)*sl1)) * math.exp(-0.5*((x[1] - ml1)/sl1)**2)

for x in x_train[y_train == -1]:
  ApErrWeight = (1 / (math.sqrt(2*math.pi)*sw_1)) * math.exp(-0.5*((x[0] - mw_1)/sw_1)**2)
  ApErrHeight = (1 / (math.sqrt(2*math.pi)*sl_1)) * math.exp(-0.5*((x[1] - ml_1)/sl_1)**2)

print('вероятность ошибки: ', BpErrWeight, BpErrHeight, ApErrWeight, ApErrHeight)

# тестовая
x = [10, 5]

classA = lambda x: -(x[0] - ml_1) ** 2 / (2 * sl_1) - (x[1] - mw_1) ** 2 / (2 * sw_1) # -1
classB = lambda x: -(x[0] - ml1) ** 2 / (2 * sl1) - (x[1] - mw1) ** 2 / (2 * sw1) # 1
y = np.argmax([classA(x), classB(x)])

print('Номер класса (0 - classA, 1 - classB): ', y)