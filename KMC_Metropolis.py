import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# 参数设置
T = 610  # 温度 (K)
k_B = 8.617e-5  # Boltzmann constant in eV/K
E_d = 1.8  # 脱附能量 (eV)
E_s = 1.58  # 吸附能量 (eV)
E_n = 0.28  # 相互作用能量 (eV)
h = 4.135667696e-15  # Planck 常数 (eV*s)

# 速率常数计算
k_d0 = 2 * k_B * T / h
k_m0 = 2 * k_B * T / h  # 假设迁移速率的预因子和脱附相同

# 系统参数设置
lattice_size = 100  # 晶格尺寸
n_categories = 5  # 最近邻的数量 (0-4)
max_steps = 50000000  # 最大模拟步数

# 初始化晶格，分子初始状态
lattice = np.zeros((lattice_size, lattice_size), dtype=int)
categories = {n: [] for n in range(n_categories)}


# 定义速率函数
def desorption_rate(n):
    return k_d0 * np.exp(-(E_d + n * E_n) / (k_B * T))


def migration_rate(n):
    return k_m0 * np.exp(-(E_s + n * E_n) / (k_B * T))


# 计算速率字典，用于不同邻居数量的速率
desorption_rates = {n: desorption_rate(n) for n in range(n_categories)}
migration_rates = {n: migration_rate(n) for n in range(n_categories)}


# 计算某个位点的邻居数
def count_neighbors(i, j):
    return sum(lattice[(i + di) % lattice_size, (j + dj) % lattice_size]
               for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)])


# 初始化所有位点的分类
for i in range(lattice_size):
    for j in range(lattice_size):
        n_neighbors = min(max(count_neighbors(i, j), 0), 4)
        categories[n_neighbors].append((i, j))


# 更新位点分类
def update_category(i, j):
    n_neighbors = min(max(count_neighbors(i, j), 0), 4)

    # 找到旧分类并移除
    for n in range(n_categories):
        if (i, j) in categories[n]:
            categories[n].remove((i, j))
            break

    # 加入新的分类
    categories[n_neighbors].append((i, j))


# 定义时间变量
time = 0

# 执行模拟
for step in range(max_steps):
    # 计算事件速率
    event_rates = {
        "desorption": sum(desorption_rates[n] * len(categories[n]) for n in range(n_categories)),
        "migration": sum(migration_rates[n] * len(categories[n]) for n in range(n_categories)),
        "adsorption": 0.33 * lattice_size * lattice_size  # 假设吸附速率为常数
    }
    total_rate = sum(event_rates.values())
    random_choice = np.random.rand() * total_rate

    # 确定事件类型
    if random_choice < event_rates["desorption"]:
        event = "desorption"
        rate_dict = desorption_rates
    elif random_choice < event_rates["desorption"] + event_rates["migration"]:
        event = "migration"
        rate_dict = migration_rates
    else:
        event = "adsorption"

    # 根据事件类型选择位点
    if event == "adsorption":
        # 随机选择一个位点进行吸附
        i, j = np.random.randint(0, lattice_size, 2)
        lattice[i, j] += 1
        update_category(i, j)  # 仅更新发生事件的位点分类
    else:
        # 随机选择邻近分类类别，然后选择位点
        n_neighbors = np.random.choice(list(rate_dict.keys()),
                                       p=[rate_dict[n] * len(categories[n]) / event_rates[event] for n in
                                          rate_dict.keys()])
        i, j = categories[n_neighbors][np.random.randint(0, len(categories[n_neighbors]))]

        # 如果事件是脱附
        if event == "desorption":
            delta_E = E_d + n_neighbors * E_n  # 计算能量差
            if lattice[i, j] > 0:
                if delta_E <= 0 or np.random.rand() < np.exp(-delta_E / (k_B * T)):
                    lattice[i, j] -= 1
                    update_category(i, j)  # 更新该位点的分类
        elif event == "migration":
            # 选择迁移方向（只向低处移动）
            directions = [(di, dj) for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                          if lattice[(i + di) % lattice_size, (j + dj) % lattice_size] < lattice[i, j]]
            if directions and lattice[i, j] > 0:
                di, dj = directions[np.random.randint(0, len(directions))]
                lattice[i, j] -= 1
                lattice[(i + di) % lattice_size, (j + dj) % lattice_size] += 1
                update_category(i, j)  # 更新原位点的分类
                update_category((i + di) % lattice_size, (j + dj) % lattice_size)  # 更新迁移后位点的分类

    # 时间步长更新
    zeta = np.random.rand()
    tau = -np.log(zeta) / total_rate
    time += tau

    # 终止条件检查
    if step >= max_steps or time >= 300:
        break
mean_value = np.mean(lattice)


print("Simulation finished.")
print("Final lattice state:\n", lattice)
print("Total simulation time:", time)
print("mean_value:", mean_value)

# 创建 x 和 y 的坐标范围
x = np.arange(lattice.shape[0])
y = np.arange(lattice.shape[1])
X, Y = np.meshgrid(x, y)

# 将 X, Y, Z 转为一维
points = np.array([X.ravel(), Y.ravel()]).T  # 原始网格的坐标点
values = lattice.ravel()  # 原始网格的值

# 定义插值后的网格
x_new = np.linspace(0, lattice.shape[0] - 1, 100)
y_new = np.linspace(0, lattice.shape[1] - 1, 100)
X_new, Y_new = np.meshgrid(x_new, y_new)
grid_z = griddata(points, values, (X_new, Y_new), method='cubic')

# 创建 3D 图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制平滑后的表面图
ax.plot_surface(X_new, Y_new, grid_z, cmap='viridis')

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()