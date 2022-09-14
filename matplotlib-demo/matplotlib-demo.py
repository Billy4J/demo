import numpy as np
from matplotlib import pyplot as plt

# 创建坐标点
x = np.linspace(-np.pi, np.pi, 255)
y1 = np.cos(x)
y2 = np.sin(x)

# 设置图片大小，像素
plt.figure(figsize=(8, 6),
           dpi=80)

# 设置子图， AxesSubplot类型
# a = plt.subplot(1, 2, 1)

# 设置子图, Axes类型
# b = plt.axes((1, 1, 1, 1))

ax = plt.gca()  # get current axes 获取当前坐标轴
ax.spines["right"].set_color('w')
ax.spines["top"].set_color("w")
ax.spines["bottom"].set_color("g")
# ac.xaxis.set_ticks_position("bottom")
ax.spines["bottom"].set_position(("data", 0))

t = 2 * np.pi / 3
plt.plot(x, np.cos(x), color="blue", linewidth=2.5, linestyle="-", label="cosine")
plt.plot(x, np.sin(x), label="sine")

plt.scatter([t, ], [np.cos(t), ], 50, color='blue')
plt.plot([t, t], [0, np.cos(t)], color='blue', linewidth=2.5, linestyle="--")
plt.annotate(r'$\sin(\frac{2\pi}{3})=\frac{\sqrt{3}}{2}$',
             xy=(t, np.cos(t)), xycoords='data',
             xytext=(+10, +30), textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

plt.legend(loc="best")
plt.xlabel("x-zhou")
plt.ylabel("y-zhou ")

# plt.xlim(-1, 1)
# plt.ylim(-0.5, 0.5)
# plt.xticks(x[::100], labels=["hello" for _ in x[::100]])  # x轴显示刻度

plt.savefig('test.png')

# 图像显示
plt.show()
