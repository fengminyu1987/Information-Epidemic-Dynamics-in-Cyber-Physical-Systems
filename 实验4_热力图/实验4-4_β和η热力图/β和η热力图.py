# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# from scipy.integrate import odeint
# import random
# import os
# import pandas as pd
# from collections import defaultdict
# import itertools
# from tqdm import tqdm
# from multiprocessing import Pool, cpu_count
# import functools
#
# # 参数设置
# initial_infected_ratio = 0.01
# num_nodes = 200
# lambda_info = 0.3  # 固定λ_info值
# lambda_star = 0.1  # 固定λ*值
# # delta = 0.4
# # mu = 0.4
# delta = 0.2
# mu = 0.2
# alpha = 1.0  # 固定α值
# seed_value = 211
# random.seed(seed_value)
# np.random.seed(seed_value)
#
# # # 参数设置
# # initial_infected_ratio = 0.01
# # num_nodes = 200
# # lambda_info = 0.1  # 固定λ_info值
# # lambda_star = 0.05  # 固定λ*值
# # delta = 0.8
# # mu = 0.4
# # alpha = 1.0  # 固定α值
# # seed_value = 211
# # random.seed(seed_value)
# # np.random.seed(seed_value)
#
# # 构建连通3-均匀超图
# def build_connected_3uniform_hypergraph(N, num_hyperedges):
#     hyperedges = []
#     nodes_included = set()
#
#     # Step 1: 初始化一个三元组
#     triplet = random.sample(range(N), 3)
#     hyperedges.append(set(triplet))
#     nodes_included.update(triplet)
#
#     # Step 2: 每个新超边包含至少一个已在图中的节点
#     while len(hyperedges) < num_hyperedges:
#         existing_node = random.choice(list(nodes_included))
#         other_nodes = random.sample([n for n in range(N) if n != existing_node], 2)
#         new_triplet = set([existing_node] + other_nodes)
#         if new_triplet not in hyperedges:
#             hyperedges.append(new_triplet)
#             nodes_included.update(new_triplet)
#
#         # 若仍未包含所有节点，则强制加入未覆盖节点
#         if len(nodes_included) < N and len(hyperedges) > N // 3:
#             uncovered = list(set(range(N)) - nodes_included)
#             for u in uncovered:
#                 partner_nodes = random.sample(list(nodes_included), 2)
#                 new_triplet = set([u] + partner_nodes)
#                 if new_triplet not in hyperedges:
#                     hyperedges.append(new_triplet)
#                     nodes_included.update(new_triplet)
#
#     return hyperedges
#
# # 预计算所有需要的数据结构
# def prepare_data_structures():
#     num_hyperedges = int(num_nodes * 2.5)
#     hyperedges = build_connected_3uniform_hypergraph(num_nodes, num_hyperedges)
#
#     # 节点到超边的映射
#     node_to_hyperedges = {i: [] for i in range(num_nodes)}
#     for idx, he in enumerate(hyperedges):
#         for node in he:
#             node_to_hyperedges[node].append(idx)
#
#     # 构建高阶传播结构 node_simplex_neighbors
#     node_simplex_neighbors = defaultdict(list)
#     for he in hyperedges:
#         for i in he:
#             others = list(set(he) - {i})
#             if len(others) == 2:
#                 node_simplex_neighbors[i].append((others[0], others[1]))
#
#     # 普通信息图
#     G_info = nx.Graph()
#     G_info.add_nodes_from(range(num_nodes))
#     p_pairwise = 0.5
#     for he in hyperedges:
#         for u, v in itertools.combinations(he, 2):
#             if random.random() < p_pairwise:
#                 G_info.add_edge(u, v)
#     a_matrix = nx.to_numpy_array(G_info)
#
#     # 物理层传播图
#     G_epidemic = nx.watts_strogatz_graph(num_nodes, 4, 0.5, seed=2)
#     b_matrix = nx.to_numpy_array(G_epidemic)
#
#     # Jaccard 权重矩阵
#     W_ij_matrix = np.zeros((num_nodes, num_nodes))
#     W_i_vector = np.zeros(num_nodes)
#     for i in range(num_nodes):
#         E_i = set(node_to_hyperedges[i])
#         for j in range(num_nodes):
#             if i == j:
#                 continue
#             E_j = set(node_to_hyperedges[j])
#             intersection = len(E_i & E_j)
#             union = len(E_i | E_j)
#             if union > 0:
#                 W_ij = (intersection / union) ** alpha
#                 W_ij_matrix[i, j] = W_ij
#                 W_i_vector[i] += W_ij
#
#     return {
#         'hyperedges': hyperedges,
#         'node_to_hyperedges': node_to_hyperedges,
#         'node_simplex_neighbors': node_simplex_neighbors,
#         'a_matrix': a_matrix,
#         'b_matrix': b_matrix,
#         'W_ij_matrix': W_ij_matrix,
#         'W_i_vector': W_i_vector
#     }
#
# # 自适应传播概率矩阵 - 修改为支持η参数
# # def get_beta_Aij_matrix(beta_U, eta, W_ij_matrix, W_i_vector):
# #     beta_Aij_matrix = np.zeros((num_nodes, num_nodes))
# #     for i in range(num_nodes):
# #         W_i = W_i_vector[i]
# #         for j in range(num_nodes):
# #             if i == j or W_i == 0:
# #                 continue
# #             gamma_ij = (W_ij_matrix[i, j] / W_i) ** eta  # 使用传入的η参数
# #             beta_Aij_matrix[i, j] = gamma_ij * beta_U
# #     return beta_Aij_matrix
#
# def get_beta_Aij_matrix(beta_U, eta, W_ij_matrix, W_i_vector, epsilon=1e-10):
#     beta_Aij_matrix = np.zeros((num_nodes, num_nodes))
#
#     for i in range(num_nodes):
#         W_i = W_i_vector[i]
#         for j in range(num_nodes):
#             if i == j or W_i == 0:
#                 continue
#             # 使用对数增强形式来计算传播概率
#             # 防止除零错误，加入一个小常数 epsilon
#             gamma_ij = np.exp(eta * np.log((W_ij_matrix[i, j] + epsilon) / (W_i + epsilon)))
#             beta_Aij_matrix[i, j] = gamma_ij * beta_U
#
#     return beta_Aij_matrix
#
# # 高阶传播 λ*
# def get_r_i_2(node_simplex_neighbors, P_A, lambda_star):
#     r_i_2 = np.ones(num_nodes)
#     for i in range(num_nodes):
#         product = 1
#         for (j, k) in node_simplex_neighbors[i]:
#             c_ijk = P_A[j] * P_A[k]
#             product *= (1 - c_ijk * lambda_star)
#         r_i_2[i] = product
#     return r_i_2
#
# # 普通信息传播
# def get_r_i_1(a_matrix, P_A, W_ij_matrix, lambda_info):
#     result = np.ones(len(P_A))
#     for i in range(len(P_A)):
#         temp = 1.0
#         for j in range(len(P_A)):
#             if a_matrix[i, j]:
#                 temp *= (1 - lambda_info * P_A[j] * W_ij_matrix[i, j])
#         result[i] = temp
#     return result
#
# # 疫情传播（在U状态时不被其邻居感染的概率）
# def get_q_i_U(b_matrix, P_AI, beta):
#     return np.prod(1 - b_matrix * P_AI[:, None] * beta, axis=0)
#
# # 疫情传播（在A状态时不被其邻居感染的概率）
# def get_q_i_A(b_matrix, P_AI, beta_Aij_matrix):
#     q_i = np.ones(num_nodes)
#     for i in range(num_nodes):
#         for j in range(num_nodes):
#             if b_matrix[j, i] == 1:
#                 q_i[i] *= (1 - P_AI[j] * beta_Aij_matrix[i, j])
#     return q_i
#
# # MMCA 迭代主体 - 修改为支持β和η参数
# def iterate_probabilities_beta_eta(beta_U, eta, data_structures):
#     # 解包数据结构
#     node_simplex_neighbors = data_structures['node_simplex_neighbors']
#     a_matrix = data_structures['a_matrix']
#     b_matrix = data_structures['b_matrix']
#     W_ij_matrix = data_structures['W_ij_matrix']
#     W_i_vector = data_structures['W_i_vector']
#
#     # 为每个进程设置不同的随机种子
#     np.random.seed(seed_value + hash((beta_U, eta)) % 1000)
#
#     P_AI = np.zeros(num_nodes)
#     P_AS = np.zeros(num_nodes)
#     P_US = np.ones(num_nodes)
#     initial_infected = np.random.choice(num_nodes, int(num_nodes * initial_infected_ratio), replace=False)
#     P_AI[initial_infected] = 1
#     P_US[initial_infected] = 0
#
#     max_iterations = 10000
#     tolerance = 1e-10
#
#     for iteration in range(max_iterations):
#         P_A = P_AI + P_AS
#         # λ_info和λ*都是固定的
#         r_i = get_r_i_1(a_matrix, P_A, W_ij_matrix, lambda_info) * get_r_i_2(node_simplex_neighbors, P_A, lambda_star)
#
#         q_i_U = get_q_i_U(b_matrix, P_AI, beta_U)
#         beta_Aij_matrix = get_beta_Aij_matrix(beta_U, eta, W_ij_matrix, W_i_vector)  # 使用传入的η参数
#         q_i_A = get_q_i_A(b_matrix, P_AI, beta_Aij_matrix)
#
#         P_US_next = P_AI * delta * mu + P_US * r_i * q_i_U + P_AS * delta * q_i_U
#         P_AS_next = P_AI * (1 - delta) * mu + P_US * (1 - r_i) * q_i_A + P_AS * (1 - delta) * q_i_A
#         P_AI_next = P_AI * (1 - mu) + P_US * ((1 - r_i) * (1 - q_i_A) + r_i * (1 - q_i_U)) + \
#                     P_AS * (delta * (1 - q_i_U) + (1 - delta) * (1 - q_i_A))
#
#         # 检查收敛
#         if (np.max(np.abs(P_US_next - P_US)) < tolerance and
#             np.max(np.abs(P_AS_next - P_AS)) < tolerance and
#             np.max(np.abs(P_AI_next - P_AI)) < tolerance):
#             break
#
#         P_US, P_AS, P_AI = P_US_next, P_AS_next, P_AI_next
#
#     infection_density = np.mean(P_AI)
#     awareness_density = np.mean(P_A)
#     return infection_density, awareness_density
#
# # 用于并行处理的包装函数
# def compute_single_point(args):
#     beta_U, eta, data_structures = args
#     return iterate_probabilities_beta_eta(beta_U, eta, data_structures)
#
# # 主计算函数
# def main():
#     print("准备数据结构...")
#     data_structures = prepare_data_structures()
#
#     # 参数网格 - 现在是β和η
#     beta_values = np.linspace(0, 1, 51)
#     eta_values = np.linspace(0, 3, 151)
#
#     # 创建参数组合
#     param_combinations = []
#     for i, beta_U in enumerate(beta_values):
#         for j, eta in enumerate(eta_values):
#             param_combinations.append((beta_U, eta, data_structures))
#
#     print(f"总共需要计算 {len(param_combinations)} 个点")
#     print(f"使用 {cpu_count()} 个CPU核心进行并行计算...")
#     print(f"固定参数: λ_info = {lambda_info}, λ* = {lambda_star}, α = {alpha}")
#
#     # 并行计算
#     with Pool(processes=cpu_count()) as pool:
#         results = list(tqdm(
#             pool.imap(compute_single_point, param_combinations),
#             total=len(param_combinations),
#             desc="计算进度"
#         ))
#
#     # 重新组织结果
#     infection_density_matrix = np.zeros((len(beta_values), len(eta_values)))
#     awareness_density_matrix = np.zeros((len(beta_values), len(eta_values)))
#
#     idx = 0
#     for i in range(len(beta_values)):
#         for j in range(len(eta_values)):
#             infection_density, awareness_density = results[idx]
#             infection_density_matrix[i, j] = infection_density
#             awareness_density_matrix[i, j] = awareness_density
#             idx += 1
#
#     # 创建输出目录
#     output_dir = r"D:/OneDrive/桌面/最新代码/实验4_热力图/实验4-4_β和η热力图"
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 构造保存路径
#     infection_path = os.path.join(output_dir, "hypergraph_infection_density_matrix_beta_eta.csv")
#     awareness_path = os.path.join(output_dir, "hypergraph_awareness_density_matrix_beta_eta.csv")
#
#     # 保存数据到 CSV 文件
#     np.savetxt(infection_path, infection_density_matrix, delimiter=",")
#     np.savetxt(awareness_path, awareness_density_matrix, delimiter=",")
#
#     print("数据已保存到 CSV 文件：")
#     print(f"感染密度矩阵: {infection_path}")
#     print(f"意识密度矩阵: {awareness_path}")
#     print("计算完成！")
#
# if __name__ == "__main__":
#     main()


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import os
import pandas as pd
from collections import defaultdict
from joblib import Parallel, delayed, cpu_count
import itertools
from tqdm import tqdm

# ------------------------ 参数设置 ------------------------
initial_infected_ratio = 0.01
num_nodes = 200
# lambda_info = 0.3  # 固定λ_info值
# lambda_star = 0.1  # 固定λ*值
lambda_info = 0.6  # 固定λ_info值
lambda_star = 0.6  # 固定λ*值
delta = 0.2
mu = 0.2
alpha = 1.0  # 固定α值
seed_value = 211
random.seed(seed_value)
np.random.seed(seed_value)


# ------------------------ 构图和传播函数 ------------------------
def build_connected_3uniform_hypergraph(N, num_hyperedges):
    """构建连通3-均匀超图"""
    hyperedges = []
    nodes_included = set()

    # 初始化一个三元组
    triplet = random.sample(range(N), 3)
    hyperedges.append(set(triplet))
    nodes_included.update(triplet)

    # 每个新超边包含至少一个已在图中的节点
    while len(hyperedges) < num_hyperedges:
        existing_node = random.choice(list(nodes_included))
        other_nodes = random.sample([n for n in range(N) if n != existing_node], 2)
        new_triplet = set([existing_node] + other_nodes)
        if new_triplet not in hyperedges:
            hyperedges.append(new_triplet)
            nodes_included.update(new_triplet)

        # 强制加入未覆盖节点
        if len(nodes_included) < N and len(hyperedges) > N // 3:
            uncovered = list(set(range(N)) - nodes_included)
            for u in uncovered:
                partner_nodes = random.sample(list(nodes_included), 2)
                new_triplet = set([u] + partner_nodes)
                if new_triplet not in hyperedges:
                    hyperedges.append(new_triplet)
                    nodes_included.update(new_triplet)

    return hyperedges


def prepare_data_structures():
    """预计算所有需要的数据结构"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    num_hyperedges = int(num_nodes * 2.5)
    hyperedges = build_connected_3uniform_hypergraph(num_nodes, num_hyperedges)

    # 节点到超边的映射
    node_to_hyperedges = {i: [] for i in range(num_nodes)}
    for idx, he in enumerate(hyperedges):
        for node in he:
            node_to_hyperedges[node].append(idx)

    # 构建高阶传播结构
    node_simplex_neighbors = defaultdict(list)
    for he in hyperedges:
        for i in he:
            others = list(set(he) - {i})
            if len(others) == 2:
                node_simplex_neighbors[i].append((others[0], others[1]))

    # 普通信息图
    G_info = nx.Graph()
    G_info.add_nodes_from(range(num_nodes))
    for he in hyperedges:
        for u, v in itertools.combinations(he, 2):
            if random.random() < 0.5:
                G_info.add_edge(u, v)
    a_matrix = nx.to_numpy_array(G_info)

    # 物理层传播图
    G_epidemic = nx.watts_strogatz_graph(num_nodes, 4, 0.5, seed=2)
    b_matrix = nx.to_numpy_array(G_epidemic)

    # Jaccard权重矩阵
    W_ij_matrix = np.zeros((num_nodes, num_nodes))
    W_i_vector = np.zeros(num_nodes)
    for i in range(num_nodes):
        E_i = set(node_to_hyperedges[i])
        for j in range(num_nodes):
            if i == j:
                continue
            E_j = set(node_to_hyperedges[j])
            inter = len(E_i & E_j)
            union = len(E_i | E_j)
            if inter > 0 and union > 0:
                w_ij = (inter / union) ** alpha
                W_ij_matrix[i, j] = w_ij
                W_i_vector[i] += w_ij

    return dict(
        node_simplex_neighbors=node_simplex_neighbors,
        a_matrix=a_matrix,
        b_matrix=b_matrix,
        W_ij_matrix=W_ij_matrix,
        W_i_vector=W_i_vector
    )


# def get_beta_Aij_matrix(beta_U, eta, W_ij_matrix, W_i_vector):
#     """计算自适应传播概率矩阵"""
#     beta_Aij = np.zeros_like(W_ij_matrix)
#     for i in range(num_nodes):
#         W_i = W_i_vector[i]
#         for j in range(num_nodes):
#             if i != j and W_i > 0:
#                 gamma_ij = (W_ij_matrix[i, j] / W_i) ** eta
#                 beta_Aij[i, j] = gamma_ij * beta_U
#     return beta_Aij

def get_beta_Aij_matrix(beta_U, eta, W_ij_matrix, W_i_vector):
    """按照论文 Eq.(3)-(4) 计算自适应传播概率矩阵"""
    beta_Aij = np.zeros_like(W_ij_matrix)

    for i in range(num_nodes):
        W_i = W_i_vector[i]

        for j in range(num_nodes):
            if i == j or W_i == 0:
                continue

            
            if eta == 0:
                
                gamma_ij = 1.0
            elif W_ij_matrix[i, j] > 0:
                gamma_ij = (W_ij_matrix[i, j] / W_i) ** eta
            else:
                gamma_ij = 0.0
            

            beta_Aij[i, j] = gamma_ij * beta_U

    return beta_Aij


def iterate_mmca(beta_U, eta):
    """MMCA迭代计算"""
    data = prepare_data_structures()
    nsn, a, b, W, Wv = data.values()

    # 设置随机种子
    np.random.seed(seed_value + hash((beta_U, eta)) % 10 ** 6)

    # 初始化状态概率
    P_AI = np.zeros(num_nodes)
    P_AS = np.zeros(num_nodes)
    P_US = np.ones(num_nodes)
    initial = np.random.choice(num_nodes, int(num_nodes * initial_infected_ratio), replace=False)
    P_AI[initial] = 1
    P_US[initial] = 0

    # 迭代计算
    for _ in range(10000):
        P_A = P_AI + P_AS

        # 计算r_i（信息传播概率）
        r_i1 = np.ones(num_nodes)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if a[i, j]:
                    r_i1[i] *= (1 - lambda_info * P_A[j] * W[i, j])

        r_i2 = np.ones(num_nodes)
        for i in range(num_nodes):
            for j, k in nsn[i]:
                r_i2[i] *= (1 - P_A[j] * P_A[k] * lambda_star)
        r_i = r_i1 * r_i2

        # 计算疫情传播概率
        q_U = np.prod(1 - b * P_AI[:, None] * beta_U, axis=0)
        beta_Aij = get_beta_Aij_matrix(beta_U, eta, W, Wv)
        q_A = np.ones(num_nodes)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if b[j, i]:
                    q_A[i] *= (1 - P_AI[j] * beta_Aij[i, j])

        # 更新状态概率
        P_US_next = P_AI * delta * mu + P_US * r_i * q_U + P_AS * delta * q_U
        P_AS_next = P_AI * (1 - delta) * mu + P_US * (1 - r_i) * q_A + P_AS * (1 - delta) * q_A
        P_AI_next = P_AI * (1 - mu) + P_US * ((1 - r_i) * (1 - q_A) + r_i * (1 - q_U)) + \
                    P_AS * (delta * (1 - q_U) + (1 - delta) * (1 - q_A))

        # 检查收敛
        if np.allclose(P_AI, P_AI_next, atol=1e-10):
            break

        P_US, P_AS, P_AI = P_US_next, P_AS_next, P_AI_next

    return np.mean(P_AI), np.mean(P_AS + P_AI)


# ------------------------ 主函数入口 ------------------------
def main():
    print("开始计算β和η热力图...")

    # 参数网格
    beta_values = np.linspace(0, 1, 51)
    eta_values = np.linspace(0, 2, 101)
    param_grid = list(itertools.product(beta_values, eta_values))

    print(f"总共需要计算 {len(param_grid)} 个 (β, η) 点")
    print(f"使用 {cpu_count()} 个CPU核心进行并行计算...")
    print(f"固定参数: λ_info = {lambda_info}, λ* = {lambda_star}, α = {alpha}")

    # 并行计算
    results = Parallel(n_jobs=64, backend='loky', verbose=10)(
        delayed(iterate_mmca)(beta, eta) for beta, eta in tqdm(param_grid)
    )

    # 重新组织结果
    infection_matrix = np.zeros((len(beta_values), len(eta_values)))
    awareness_matrix = np.zeros((len(beta_values), len(eta_values)))

    for idx, (inf, aware) in enumerate(results):
        i = idx // len(eta_values)
        j = idx % len(eta_values)
        infection_matrix[i, j] = inf
        awareness_matrix[i, j] = aware

    # 保存结果
    out_dir = r"D:\OneDrive\桌面\最新代码\实验4_热力图\实验4-4_β和η热力图"
    os.makedirs(out_dir, exist_ok=True)

    infection_path = os.path.join(out_dir, "hypergraph_infection_density_matrix_beta_eta.csv")
    awareness_path = os.path.join(out_dir, "hypergraph_awareness_density_matrix_beta_eta.csv")

    np.savetxt(infection_path, infection_matrix, delimiter=",")
    np.savetxt(awareness_path, awareness_matrix, delimiter=",")

    print("数据已保存到 CSV 文件：")
    print(f"感染密度矩阵: {infection_path}")
    print(f"意识密度矩阵: {awareness_path}")
    print("计算完成！")


if __name__ == "__main__":
    main()