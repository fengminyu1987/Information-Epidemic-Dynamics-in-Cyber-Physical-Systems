import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import odeint
import random
import os
import pandas as pd
from collections import defaultdict
import itertools
from tqdm import tqdm

# # --- 参数设置 ---
# initial_infected_ratio = 0.01
# num_nodes = 200
# lambda_info = 0.5
# lambda_star = 0.8
# delta = 0.4
# mu = 0.4
# alpha_values = [0.0, 0.5, 1.0, 2.0]
# eta = 1.0
# seed_value = 211
# random.seed(seed_value)
# np.random.seed(seed_value)

# 参数设置
initial_infected_ratio = 0.01
num_nodes = 200
lambda_info = 0.3  # 固定λ_info值
lambda_star = 0.1  # 固定λ*值
delta = 0.2
mu = 0.2
# lambda_info = 0.2  # 固定λ_info值
# lambda_star = 0.1  # 固定λ*值
# delta = 0.2
# mu = 0.2
eta = 1.0
alpha_values = [0.0, 0.5, 1.0, 2.0]
seed_value = 211
random.seed(seed_value)
np.random.seed(seed_value)

# --- 构建连通3-均匀超图 ---
def build_connected_3uniform_hypergraph(N, num_hyperedges):
    hyperedges = []
    nodes_included = set()

    # Step 1: 初始化一个三元组
    triplet = random.sample(range(N), 3)
    hyperedges.append(set(triplet))
    nodes_included.update(triplet)

    # Step 2: 每个新超边包含至少一个已在图中的节点
    while len(hyperedges) < num_hyperedges:
        existing_node = random.choice(list(nodes_included))
        other_nodes = random.sample([n for n in range(N) if n != existing_node], 2)
        new_triplet = set([existing_node] + other_nodes)
        if new_triplet not in hyperedges:
            hyperedges.append(new_triplet)
            nodes_included.update(new_triplet)

        # 若仍未包含所有节点，则强制加入未覆盖节点
        if len(nodes_included) < N and len(hyperedges) > N // 3:
            uncovered = list(set(range(N)) - nodes_included)
            for u in uncovered:
                partner_nodes = random.sample(list(nodes_included), 2)
                new_triplet = set([u] + partner_nodes)
                if new_triplet not in hyperedges:
                    hyperedges.append(new_triplet)
                    nodes_included.update(new_triplet)

    return hyperedges

num_hyperedges = int(num_nodes * 2.5)
hyperedges = build_connected_3uniform_hypergraph(num_nodes, num_hyperedges)

# --- 改进的超图构造函数 ---
# def build_denser_hypergraph(N, num_hyperedges, hyperedge_size=3, attachment_preference=0.8):
#     """
#     构建一个连接更密集、节点共享超边可能性更高的超图。
#     Args:
#         N (int): 节点数量。
#         num_hyperedges (int): 超边数量。
#         hyperedge_size (int): 每个超边的节点数量 (默认为3，对应3-均匀超图)。
#         attachment_preference (float): 偏好依附的强度 (0-1)。
#                                         1表示完全偏好高度连接的节点，0表示完全随机。
#                                         更高的值会增加节点间共享超边的可能性。
#     Returns:
#         list: 包含超边的列表，每个超边是一个set。
#     """
#     hyperedges = []
#     node_degrees = defaultdict(int)  # 记录节点的度 (参与的超边数量)
#
#     # 确保至少有一个初始超边，且包含足够多的节点以启动连接
#     if N < hyperedge_size:
#         raise ValueError("节点数量必须大于或等于超边大小。")
#
#     # Step 1: 初始化第一个超边
#     initial_nodes = random.sample(range(N), hyperedge_size)
#     new_he = set(initial_nodes)
#     hyperedges.append(new_he)
#     for node in new_he:
#         node_degrees[node] += 1
#
#     # Step 2: 迭代添加剩余超边
#     # 目标是增加节点间共享超边的可能性
#     for _ in tqdm(range(len(hyperedges), num_hyperedges), desc="Building Hypergraph"):
#         current_nodes_in_graph = list(node_degrees.keys())
#
#         # 偏好依附策略: 优先选择度高的节点
#         # 创建一个基于度的节点选择权重列表
#         if current_nodes_in_graph and sum(node_degrees.values()) > 0:
#             nodes_weights = [node_degrees[node] for node in current_nodes_in_graph]
#             # 归一化权重
#             total_weight = sum(nodes_weights)
#             if total_weight == 0:  # 避免除以零
#                 nodes_weights = [1] * len(current_nodes_in_graph)  # 如果所有度都为0，则平均选择
#
#             # 根据权重选择一个节点作为新超边的“锚点”
#             anchor_node = random.choices(current_nodes_in_graph, weights=nodes_weights, k=1)[0]
#         else:  # 如果还没有节点被添加到超边中 (仅在初始步骤可能发生)
#             anchor_node = random.choice(range(N))
#
#         new_he = {anchor_node}
#
#         # 选择其余 hyperedge_size - 1 个节点
#         remaining_nodes_pool = [n for n in range(N) if n not in new_he]
#
#         # 尝试从已存在于图中的节点中选择，以增加重叠
#         # 使用偏好依附的概率决定是否从已存在的节点中选择
#         if random.random() < attachment_preference and len(current_nodes_in_graph) >= (hyperedge_size - 1):
#             # 尝试从当前已连接的节点中选择剩余节点，增加共享度
#             try_nodes = [n for n in current_nodes_in_graph if n not in new_he]
#             if len(try_nodes) >= (hyperedge_size - 1):
#                 new_members = random.sample(try_nodes, hyperedge_size - 1)
#             else:  # 如果不够，则从所有节点中随机选择
#                 new_members = random.sample(remaining_nodes_pool, hyperedge_size - 1)
#         else:  # 随机选择
#             if len(remaining_nodes_pool) >= (hyperedge_size - 1):
#                 new_members = random.sample(remaining_nodes_pool, hyperedge_size - 1)
#             else:  # 如果剩余节点不够，则允许重复选择（如果允许的话，这里不鼓励，但为了防止死循环）
#                 # 或者退化为更小的超边，但这里保持 hyperedge_size
#                 # 理论上，在 N 足够大的情况下不会发生
#                 print(f"Warning: Not enough unique nodes for hyperedge {len(hyperedges) + 1}. Re-sampling.")
#                 new_members = random.sample(range(N), hyperedge_size - 1)  # Fallback to any node
#
#         new_he.update(new_members)
#
#         # 确保超边是新的
#         if new_he not in hyperedges and len(new_he) == hyperedge_size:  # 确保超边大小正确
#             hyperedges.append(new_he)
#             for node in new_he:
#                 node_degrees[node] += 1
#         elif len(new_he) != hyperedge_size:  # 超边大小不正确，跳过
#             continue
#         else:  # 如果是重复超边，尝试重新生成
#             # 这里可以加一个重试机制，但为了简化，直接跳过并让循环继续
#             pass
#
#     # 最终检查和确保所有节点至少属于一个超边 (可选，但推荐用于连通性)
#     # 如果有孤立节点，强制将其连接到现有超边
#     all_nodes_in_hypergraph = set(node for he in hyperedges for node in he)
#     if len(all_nodes_in_hypergraph) < N:
#         uncovered_nodes = list(set(range(N)) - all_nodes_in_hypergraph)
#         print(f"Warning: {len(uncovered_nodes)} nodes are not covered. Forcing connection.")
#         for u_node in uncovered_nodes:
#             # 找到一个现有的超边，并尝试将此节点添加进去（打破均匀性，但确保连通）
#             # 或者创建一个新的超边包含它和两个随机现有节点
#             if len(hyperedges) > 0:
#                 # 尝试从现有超边中随机选择一个并添加，或者创建新超边
#                 # 这里为了保持k-均匀性，我们创建新的超边
#                 partner_nodes = random.sample(list(all_nodes_in_hypergraph), hyperedge_size - 1)
#                 new_he = set([u_node] + partner_nodes)
#                 if new_he not in hyperedges:
#                     hyperedges.append(new_he)
#                     for node in new_he:
#                         node_degrees[node] += 1
#             else:  # 极端情况，如果一个超边都没有
#                 new_he = set(random.sample(range(N), hyperedge_size))
#                 hyperedges.append(new_he)
#                 for node in new_he:
#                     node_degrees[node] += 1
#
#     return hyperedges
#
#
# # --- 在主运行逻辑中调用新的超图构造函数 ---
# # 增加超边数量，以增加节点重叠的可能性
# # 原始: num_hyperedges = int(num_nodes * 1.2) (240个超边)
# # 尝试增加到 num_nodes * 2 或更高
# num_hyperedges_new = int(num_nodes * 2.5)  # 尝试更大的超边数量
# #num_hyperedges_new = int(num_nodes / 3 * math.log(num_nodes))
# hyperedges = build_denser_hypergraph(num_nodes, num_hyperedges_new, hyperedge_size=3, attachment_preference=0.7)


# --- 节点到超边的映射 ---
node_to_hyperedges = {i: [] for i in range(num_nodes)}
for idx, he in enumerate(hyperedges):
    for node in he:
        node_to_hyperedges[node].append(idx)

# --- 构建高阶传播结构 node_simplex_neighbors ---
node_simplex_neighbors = defaultdict(list)
for he in hyperedges:
    for i in he:
        others = list(set(he) - {i})
        if len(others) == 2:
            node_simplex_neighbors[i].append((others[0], others[1]))

# --- 普通信息图 ---
G_info = nx.Graph()
G_info.add_nodes_from(range(num_nodes))
p_pairwise = 0.5
for he in hyperedges:
    for u, v in itertools.combinations(he, 2):
        if random.random() < p_pairwise:
            G_info.add_edge(u, v)
a_matrix = nx.to_numpy_array(G_info)

# --- 物理层传播图 ---
G_epidemic = nx.watts_strogatz_graph(num_nodes, 4, 0.5, seed=2)
b_matrix = nx.to_numpy_array(G_epidemic)


# --- 改进的Jaccard权重矩阵计算 ---
def compute_weights_with_alpha(alpha):
    W_ij_matrix = np.zeros((num_nodes, num_nodes))
    W_i_vector = np.zeros(num_nodes)

    # 先计算原始Jaccard系数
    jaccard_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        E_i = set(node_to_hyperedges[i])
        for j in range(num_nodes):
            if i == j:
                continue
            E_j = set(node_to_hyperedges[j])
            intersection = len(E_i & E_j)
            union = len(E_i | E_j)
            if union > 0:
                jaccard_matrix[i, j] = intersection / union

    # 根据α值调整权重计算方式
    if alpha == 0:
        # α=0时，所有非零权重设为1（均匀权重）
        W_ij_matrix = (jaccard_matrix > 0).astype(float)
    else:
        # α>0时，使用幂次调节
        
        W_ij_matrix = np.power(jaccard_matrix, alpha)

    # 计算每个节点的权重总和
    for i in range(num_nodes):
        W_i_vector[i] = np.sum(W_ij_matrix[i, :])

    return W_ij_matrix, W_i_vector


# 计算权重矩阵
#W_ij_matrix, W_i_vector = compute_weights_with_alpha(alpha)
#W_ij_matrix, W_i_vector = compute_weights_with_alpha(alpha_values)

# --- 自适应传播概率矩阵 ---
def get_beta_Aij_matrix(beta_U, eta, W_ij_matrix, W_i_vector):
    beta_Aij_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        W_i = W_i_vector[i]
        for j in range(num_nodes):
            if i == j or W_i == 0:
                continue
            # 增强η的影响
            gamma_ij = (W_ij_matrix[i, j] / W_i) ** eta
            beta_Aij_matrix[i, j] = gamma_ij * beta_U
    return beta_Aij_matrix


# --- 高阶传播 λ* ---
def get_r_i_2(node_simplex_neighbors, P_A, lambda_star):
    r_i_2 = np.ones(num_nodes)
    for i in range(num_nodes):
        product = 1
        for (j, k) in node_simplex_neighbors[i]:
            c_ijk = P_A[j] * P_A[k]
            product *= (1 - c_ijk * lambda_star)
        r_i_2[i] = product
    return r_i_2


# --- 改进的普通信息传播 ---
def get_r_i_1(a_matrix, P_A, W_ij_matrix, lambda_info):
    result = np.ones(len(P_A))
    for i in range(len(P_A)):
        temp = 1.0
        for j in range(len(P_A)):
            if a_matrix[i, j] > 0:  # 存在连接
                # 使用权重调节传播强度
                weighted_lambda = lambda_info * W_ij_matrix[i, j]
                temp *= (1 - weighted_lambda * P_A[j])
        result[i] = temp
    return result


# --- 疫情传播（在U状态时不被其邻居感染的概率）---
def get_q_i_U(b_matrix, P_AI, beta):
    return np.prod(1 - b_matrix * P_AI[:, None] * beta, axis=0)


# --- 疫情传播（在A状态时不被其邻居感染的概率）---
def get_q_i_A(b_matrix, P_AI, beta_Aij_matrix):
    q_i = np.ones(num_nodes)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if b_matrix[j, i] == 1:
                q_i[i] *= (1 - P_AI[j] * beta_Aij_matrix[i, j])
    return q_i


# --- MMCA 迭代主体 ---
def iterate_probabilities_adaptive_beta(beta_U, eta, alpha_val):
    # 重新计算权重矩阵
    W_ij_matrix_local, W_i_vector_local = compute_weights_with_alpha(alpha_val)

    P_AI = np.zeros(num_nodes)
    P_AS = np.zeros(num_nodes)
    P_US = np.ones(num_nodes)
    initial_infected = np.random.choice(num_nodes, int(num_nodes * initial_infected_ratio), replace=False)
    P_AI[initial_infected] = 1
    P_US[initial_infected] = 0

    max_iterations = 10000
    tolerance = 1e-10

    for iteration in range(max_iterations):
        P_A = P_AI + P_AS

        # 使用当前的权重矩阵
        r_i = (get_r_i_1(a_matrix, P_A, W_ij_matrix_local, lambda_info) *
               get_r_i_2(node_simplex_neighbors, P_A, lambda_star))

        q_i_U = get_q_i_U(b_matrix, P_AI, beta_U)
        beta_Aij_matrix = get_beta_Aij_matrix(beta_U, eta, W_ij_matrix_local, W_i_vector_local)
        q_i_A = get_q_i_A(b_matrix, P_AI, beta_Aij_matrix)

        P_US_next = P_AI * delta * mu + P_US * r_i * q_i_U + P_AS * delta * q_i_U
        P_AS_next = P_AI * (1 - delta) * mu + P_US * (1 - r_i) * q_i_A + P_AS * (1 - delta) * q_i_A
        P_AI_next = P_AI * (1 - mu) + P_US * ((1 - r_i) * (1 - q_i_A) + r_i * (1 - q_i_U)) + \
                    P_AS * (delta * (1 - q_i_U) + (1 - delta) * (1 - q_i_A))

        # 检查收敛
        if (np.max(np.abs(P_US_next - P_US)) < tolerance and
                np.max(np.abs(P_AS_next - P_AS)) < tolerance and
                np.max(np.abs(P_AI_next - P_AI)) < tolerance):
            break

        P_US, P_AS, P_AI = P_US_next, P_AS_next, P_AI_next

    infection_density = np.mean(P_AI)
    awareness_density = np.mean(P_A)
    return infection_density, awareness_density


# --- 测试不同α值的影响 ---
def test_alpha_effect():
    alpha_values = [0.0, 0.5, 1.0, 2.0]
    beta_U_values = np.linspace(0, 1, 51)

    results = {}

    for alpha_val in alpha_values:
        print(f"Testing α = {alpha_val}")
        infection_densities = []
        awareness_densities = []

        for beta_U in tqdm(beta_U_values, desc=f"α={alpha_val}"):
            infection_density, awareness_density = iterate_probabilities_adaptive_beta(beta_U, eta, alpha_val)
            infection_densities.append(infection_density)
            awareness_densities.append(awareness_density)

        results[alpha_val] = {
            'beta_U': beta_U_values,
            'infection_density': infection_densities,
            'awareness_density': awareness_densities
        }

    return results




# 设置默认字体为 Times New Roman
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'


# 运行测试
results = test_alpha_effect()
colors = ['blue', 'green', 'red', 'orange']
alpha_values = [0.0, 0.5, 1.0, 2.0]  # 假设的alpha值，请根据实际情况调整

plt.figure(figsize=(10, 6))
# 感知密度图
for i, alpha_val in enumerate(alpha_values):
    plt.plot(results[alpha_val]['beta_U'], results[alpha_val]['awareness_density'],
             color=colors[i], marker='o', linewidth=4, markersize=8,
             label=f'α = {alpha_val}')

# plt.title('Awareness Density vs. β for Different α Values')
plt.xlabel(r'$\beta$', fontsize=21)
plt.ylabel('Awareness Density', fontsize=21)
plt.xticks(fontsize=19)  # 增大x轴刻度字体大小
plt.yticks(fontsize=19)  # 增大y轴刻度字体大小
plt.grid(True, alpha=0.8)
plt.legend(fontsize=24)
plt.tight_layout()
save_dir = r'D:\第一篇论文\实验代码\最新代码（完整版）\实验3_α和η的影响\α的影响'
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, f'感知密度随α变化图.pdf'), format='pdf')
plt.savefig(os.path.join(save_dir, f'感知密度随α变化图.png'), dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
# 感染密度图
# plt.figure(figsize=(12, 7))
for i, alpha_val in enumerate(alpha_values):
    plt.plot(results[alpha_val]['beta_U'], results[alpha_val]['infection_density'],
             color=colors[i], marker='s', linewidth=4, markersize=8,
             label=f'α = {alpha_val}')

# plt.title('Infection Density vs. β for Different α Values')
plt.xlabel(r'$\beta$', fontsize=21)
plt.ylabel('Infection Density', fontsize=21)
plt.xticks(fontsize=19)  # 增大x轴刻度字体大小
plt.yticks(fontsize=19)  # 增大y轴刻度字体大小
plt.grid(True, alpha=0.8)
# plt.legend(fontsize=22)
plt.legend(loc='lower right', fontsize=24)  # 图例显示在右下角，字号可自调
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f'感染密度随α变化图.pdf'), format='pdf')
plt.savefig(os.path.join(save_dir, f'感染密度随α变化图.png'), dpi=300, bbox_inches='tight')
plt.show()

# 保存结果
for alpha_val in alpha_values:
    df = pd.DataFrame({
        'beta_U': results[alpha_val]['beta_U'],
        'infection_density': results[alpha_val]['infection_density'],
        'awareness_density': results[alpha_val]['awareness_density']
    })

    filename = os.path.join(save_dir, f'MMCA_α={alpha_val}.csv')
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


# import numpy as np
# import networkx as nx
# import pandas as pd
# from collections import defaultdict
# import itertools
# from tqdm import tqdm
# import random
# import os

# # 参数设置
# initial_infected_ratio = 0.01
# num_nodes = 200
# lambda_info = 0.3  # 固定λ_info值
# lambda_star = 0.1  # 固定λ*值
# delta = 0.2
# mu = 0.2
# eta = 1.0
# alpha_values = [0.0, 0.5, 1.0, 2.0]
# seed_value = 211
# random.seed(seed_value)
# np.random.seed(seed_value)

# # --- 构建连通3-均匀超图 ---
# def build_connected_3uniform_hypergraph(N, num_hyperedges):
#     hyperedges = []
#     nodes_included = set()

#     # Step 1: 初始化一个三元组
#     triplet = random.sample(range(N), 3)
#     hyperedges.append(set(triplet))
#     nodes_included.update(triplet)

#     # Step 2: 每个新超边包含至少一个已在图中的节点
#     while len(hyperedges) < num_hyperedges:
#         existing_node = random.choice(list(nodes_included))
#         other_nodes = random.sample([n for n in range(N) if n != existing_node], 2)
#         new_triplet = set([existing_node] + other_nodes)
#         if new_triplet not in hyperedges:
#             hyperedges.append(new_triplet)
#             nodes_included.update(new_triplet)

#         # 若仍未包含所有节点，则强制加入未覆盖节点
#         if len(nodes_included) < N and len(hyperedges) > N // 3:
#             uncovered = list(set(range(N)) - nodes_included)
#             for u in uncovered:
#                 partner_nodes = random.sample(list(nodes_included), 2)
#                 new_triplet = set([u] + partner_nodes)
#                 if new_triplet not in hyperedges:
#                     hyperedges.append(new_triplet)
#                     nodes_included.update(new_triplet)

#     return hyperedges

# num_hyperedges = int(num_nodes * 2.5)
# hyperedges = build_connected_3uniform_hypergraph(num_nodes, num_hyperedges)

# # --- 节点到超边的映射 ---
# node_to_hyperedges = {i: [] for i in range(num_nodes)}
# for idx, he in enumerate(hyperedges):
#     for node in he:
#         node_to_hyperedges[node].append(idx)

# # --- 构建高阶传播结构 node_simplex_neighbors ---
# node_simplex_neighbors = defaultdict(list)
# for he in hyperedges:
#     for i in he:
#         others = list(set(he) - {i})
#         if len(others) == 2:
#             node_simplex_neighbors[i].append((others[0], others[1]))

# # --- 普通信息图 ---
# G_info = nx.Graph()
# G_info.add_nodes_from(range(num_nodes))
# p_pairwise = 0.5
# for he in hyperedges:
#     for u, v in itertools.combinations(he, 2):
#         if random.random() < p_pairwise:
#             G_info.add_edge(u, v)
# a_matrix = nx.to_numpy_array(G_info)

# # --- 物理层传播图 ---
# G_epidemic = nx.watts_strogatz_graph(num_nodes, 4, 0.5, seed=2)
# b_matrix = nx.to_numpy_array(G_epidemic)

# # --- 改进的Jaccard权重矩阵计算 ---
# def compute_weights_with_alpha(alpha):
#     W_ij_matrix = np.zeros((num_nodes, num_nodes))
#     W_i_vector = np.zeros(num_nodes)

#     # 先计算原始Jaccard系数
#     jaccard_matrix = np.zeros((num_nodes, num_nodes))
#     for i in range(num_nodes):
#         E_i = set(node_to_hyperedges[i])
#         for j in range(num_nodes):
#             if i == j:
#                 continue
#             E_j = set(node_to_hyperedges[j])
#             intersection = len(E_i & E_j)
#             union = len(E_i | E_j)
#             if union > 0:
#                 jaccard_matrix[i, j] = intersection / union

#     # 根据α值调整权重计算方式
#     if alpha == 0:
#         # α=0时，所有非零权重设为1（均匀权重）
#         W_ij_matrix = (jaccard_matrix > 0).astype(float)
#     else:
#         # α>0时，使用幂次调节
#         W_ij_matrix = np.power(jaccard_matrix, alpha)

#     # 计算每个节点的权重总和
#     for i in range(num_nodes):
#         W_i_vector[i] = np.sum(W_ij_matrix[i, :])

#     return W_ij_matrix, W_i_vector

# # --- 自适应传播概率矩阵 ---
# def get_beta_Aij_matrix(beta_U, eta, W_ij_matrix, W_i_vector):
#     beta_Aij_matrix = np.zeros((num_nodes, num_nodes))
#     for i in range(num_nodes):
#         W_i = W_i_vector[i]
#         for j in range(num_nodes):
#             if i == j or W_i == 0:
#                 continue
#             # 增强η的影响
#             gamma_ij = (W_ij_matrix[i, j] / W_i) ** eta
#             beta_Aij_matrix[i, j] = gamma_ij * beta_U
#     return beta_Aij_matrix

# # --- 高阶传播 λ* ---
# def get_r_i_2(node_simplex_neighbors, P_A, lambda_star):
#     r_i_2 = np.ones(num_nodes)
#     for i in range(num_nodes):
#         product = 1
#         for (j, k) in node_simplex_neighbors[i]:
#             c_ijk = P_A[j] * P_A[k]
#             product *= (1 - c_ijk * lambda_star)
#         r_i_2[i] = product
#     return r_i_2

# # --- 改进的普通信息传播 ---
# def get_r_i_1(a_matrix, P_A, W_ij_matrix, lambda_info):
#     result = np.ones(len(P_A))
#     for i in range(len(P_A)):
#         temp = 1.0
#         for j in range(len(P_A)):
#             if a_matrix[i, j] > 0:  # 存在连接
#                 # 使用权重调节传播强度
#                 weighted_lambda = lambda_info * W_ij_matrix[i, j]
#                 temp *= (1 - weighted_lambda * P_A[j])
#         result[i] = temp
#     return result

# # --- 疫情传播（在U状态时不被其邻居感染的概率）---
# def get_q_i_U(b_matrix, P_AI, beta):
#     return np.prod(1 - b_matrix * P_AI[:, None] * beta, axis=0)

# # --- 疫情传播（在A状态时不被其邻居感染的概率）---
# def get_q_i_A(b_matrix, P_AI, beta_Aij_matrix):
#     q_i = np.ones(num_nodes)
#     for i in range(num_nodes):
#         for j in range(num_nodes):
#             if b_matrix[j, i] == 1:
#                 q_i[i] *= (1 - P_AI[j] * beta_Aij_matrix[i, j])
#     return q_i

# # --- MMCA 迭代主体 ---
# def iterate_probabilities_adaptive_beta(beta_U, eta, alpha_val):
#     # 重新计算权重矩阵
#     W_ij_matrix_local, W_i_vector_local = compute_weights_with_alpha(alpha_val)

#     P_AI = np.zeros(num_nodes)
#     P_AS = np.zeros(num_nodes)
#     P_US = np.ones(num_nodes)
#     initial_infected = np.random.choice(num_nodes, int(num_nodes * initial_infected_ratio), replace=False)
#     P_AI[initial_infected] = 1
#     P_US[initial_infected] = 0

#     max_iterations = 10000
#     tolerance = 1e-10

#     for iteration in range(max_iterations):
#         P_A = P_AI + P_AS

#         # 使用当前的权重矩阵
#         r_i = (get_r_i_1(a_matrix, P_A, W_ij_matrix_local, lambda_info) *
#                get_r_i_2(node_simplex_neighbors, P_A, lambda_star))

#         q_i_U = get_q_i_U(b_matrix, P_AI, beta_U)
#         beta_Aij_matrix = get_beta_Aij_matrix(beta_U, eta, W_ij_matrix_local, W_i_vector_local)
#         q_i_A = get_q_i_A(b_matrix, P_AI, beta_Aij_matrix)

#         P_US_next = P_AI * delta * mu + P_US * r_i * q_i_U + P_AS * delta * q_i_U
#         P_AS_next = P_AI * (1 - delta) * mu + P_US * (1 - r_i) * q_i_A + P_AS * (1 - delta) * q_i_A
#         P_AI_next = P_AI * (1 - mu) + P_US * ((1 - r_i) * (1 - q_i_A) + r_i * (1 - q_i_U)) + \
#                     P_AS * (delta * (1 - q_i_U) + (1 - delta) * (1 - q_i_A))

#         # 检查收敛
#         if (np.max(np.abs(P_US_next - P_US)) < tolerance and
#                 np.max(np.abs(P_AS_next - P_AS)) < tolerance and
#                 np.max(np.abs(P_AI_next - P_AI)) < tolerance):
#             break

#         P_US, P_AS, P_AI = P_US_next, P_AS_next, P_AI_next

#     infection_density = np.mean(P_AI)
#     awareness_density = np.mean(P_A)
#     return infection_density, awareness_density

# # --- 测试不同α值的影响 ---
# def test_alpha_effect():
#     alpha_values = [0.0, 0.5, 1.0, 2.0]
#     beta_U_values = np.linspace(0, 1, 51)

#     results = {}

#     for alpha_val in alpha_values:
#         print(f"Testing α = {alpha_val}")
#         infection_densities = []
#         awareness_densities = []

#         for beta_U in tqdm(beta_U_values, desc=f"α={alpha_val}"):
#             infection_density, awareness_density = iterate_probabilities_adaptive_beta(beta_U, eta, alpha_val)
#             infection_densities.append(infection_density)
#             awareness_densities.append(awareness_density)

#         results[alpha_val] = {
#             'beta_U': beta_U_values,
#             'infection_density': infection_densities,
#             'awareness_density': awareness_densities
#         }

#     return results

# # 运行测试并保存结果
# if __name__ == "__main__":
#     results = test_alpha_effect()
    
#     # 创建保存目录
#     save_dir = r'D:\OneDrive\桌面\最新代码\实验3_α和η的影响\α的影响'
#     os.makedirs(save_dir, exist_ok=True)
    
#     # 保存结果到CSV文件
#     for alpha_val in alpha_values:
#         df = pd.DataFrame({
#             'beta_U': results[alpha_val]['beta_U'],
#             'infection_density': results[alpha_val]['infection_density'],
#             'awareness_density': results[alpha_val]['awareness_density']
#         })

#         filename = os.path.join(save_dir, f'MMCA_α={alpha_val}.csv')
#         df.to_csv(filename, index=False)
#         print(f"Results saved to {filename}")
    
#     print("所有数据生成完成！")