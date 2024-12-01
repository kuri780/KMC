import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# 参数设置
T = 710  # 温度 (K)
k_B = 8.617e-5  # Boltzmann constant in eV/K
E_d = 1.8  # 脱附能量 (eV)
E_s = 1.58 # 迁移能量 (eV)
E_n = 0.28  # 相互作用能量 (eV)
h = 4.135667696e-15  # Planck 常数 (eV*s)

# 速率常数计算
k_d0 = 2 * k_B * T / h
k_m0 = 2 * k_B * T / h  # 假设迁移速率的预因子和脱附相同

# 系统参数设置
lattice_size = 300  # 晶格尺寸
n_categories = 5  # 最近邻的数量 (0-4)
max_steps = 100000000  # 最大模拟步数

# 初始化晶格，分子初始状态
lattice = np.zeros((lattice_size, lattice_size), dtype=int)
categories = {n: set() for n in range(n_categories)}  # 使用集合来管理类别

# 定义速率函数
def desorption_rate(n):
    return k_d0 * np.exp(-(E_d + n * E_n) / (k_B * T))

def migration_rate(n):
    return k_d0 * np.exp(-(E_s + n * E_n) / (k_B * T))

# 计算速率字典，用于不同邻居数量的速率
desorption_rates = {n: desorption_rate(n) for n in range(n_categories)}
migration_rates = {n: migration_rate(n) for n in range(n_categories)}

# 计算某个位点的邻居数
def count_neighbors(i, j):
    """计算位点 (i, j) 的邻居数量，邻居的高度需要大于或等于该点的高度才计入。"""
    current_height = lattice[i, j]
    return sum(
        1 for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if lattice[(i + di) % lattice_size, (j + dj) % lattice_size] >= current_height
    )

# 初始化所有位点的分类
for i in range(lattice_size):
    for j in range(lattice_size):
        n_neighbors = count_neighbors(i, j)
        categories[n_neighbors].add((i, j))

# 更新位点分类
def update_category(i, j):
    """更新发生事件的位点的邻居分类。"""
    n_neighbors = count_neighbors(i, j)

    # 查找并移除旧分类
    for n in range(n_categories):
        if (i, j) in categories[n]:
            categories[n].remove((i, j))
            break

    # 将位点添加到新分类中
    categories[n_neighbors].add((i, j))

# 定义时间变量
time = 0
time_data = []
roughness_data = []
# 执行模拟
for step in range(max_steps):
    # 计算事件速率
    event_rates = {
        "desorption": sum(desorption_rates[n] * len(categories[n]) for n in range(n_categories)),
        "migration": sum(migration_rates[n] * len(categories[n]) for n in range(n_categories)),
        "adsorption": 1.0 * lattice_size * lattice_size  # 假设吸附速率为常数
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
        update_category(i, j)  # 更新该点的分类

        # 更新邻居点的分类
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = (i + di) % lattice_size, (j + dj) % lattice_size
            update_category(ni, nj)  # 更新邻居点的分类
    else:
        # 随机选择邻近分类类别，然后选择位点
        n_neighbors = np.random.choice(list(rate_dict.keys()),
                                       p=[rate_dict[n] * len(categories[n]) / event_rates[event] for n in
                                          rate_dict.keys()])
        if categories[n_neighbors]:  # 检查该类别是否存在可用位点
            i, j = categories[n_neighbors].pop()
            categories[n_neighbors].add((i, j))  # 暂时保持位置，防止空分类

            # 执行脱附或迁移事件
            if event == "desorption" and lattice[i, j] > 0:
                lattice[i, j] -= 1
                update_category(i, j)  # 更新当前点的分类

                # 更新邻居点的分类
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = (i + di) % lattice_size, (j + dj) % lattice_size
                    update_category(ni, nj)

            elif event == "migration" and lattice[i, j] > 0:
                # 选择迁移方向（只向低处移动）
                directions = [(di, dj) for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                              if lattice[(i + di) % lattice_size, (j + dj) % lattice_size] < lattice[i, j]]
                if directions:
                    di, dj = directions[np.random.randint(0, len(directions))]
                    ni, nj = (i + di) % lattice_size, (j + dj) % lattice_size

                    lattice[i, j] -= 1
                    lattice[ni, nj] += 1
                    update_category(i, j)  # 更新原位置的分类
                    update_category(ni, nj)  # 更新迁移后位置的分类

                    # 更新周围邻居的分类
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        neighbor_i, neighbor_j = (i + di) % lattice_size, (j + dj) % lattice_size
                        update_category(neighbor_i, neighbor_j)
                        neighbor_i, neighbor_j = (ni + di) % lattice_size, (nj + dj) % lattice_size
                        update_category(neighbor_i, neighbor_j)

    # 时间步长更新
    time += -np.log(np.random.rand()) / total_rate
    # 每5万步记录一次粗糙度
    if step % 50000 == 0:
        mean_height = np.mean(lattice)
        rms_roughness = np.sqrt(np.mean((lattice - mean_height) ** 2))
        time_data.append(time)
        roughness_data.append(rms_roughness)
        print(f"Step: {step}, Time: {time:.2f}, RMS Roughness: {rms_roughness:.4f}")

    # 终止条件检查
    if step >= max_steps or time >= 100:
        break

# 计算统计结果
mean_height = np.mean(lattice)
rms_roughness = np.sqrt(np.mean((lattice - mean_height)**2))


print("Simulation finished.")
print("Total simulation time:", time)
print("Mean Value:", mean_height)
print("RMS Roughness:", rms_roughness)

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

plt.figure()
plt.plot(time_data, roughness_data, label="RMS Roughness over Time")
plt.xlabel("Time")
plt.ylabel("RMS Roughness")
plt.title("Roughness vs. Time")
plt.legend()

plt.show()
