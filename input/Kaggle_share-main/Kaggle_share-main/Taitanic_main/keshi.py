import networkx as nx
import matplotlib.pyplot as plt
# 初期化
G = nx.Graph()
# エッジを定義する(頂点２つとその間の重みを指定しています (頂点１,頂点2、重み))
edge_list = [(0, 1, 5), (0, 2, 1), (1, 2, 3), (1, 3, 1), (3, 4, 2),
             (4, 2, 2), (3, 6, 1), (4, 6, 5), (2, 5, 2), (4, 5, 3)]
# エッジのラベルを重みで定義
edge_labels = {}
for i in range(len(edge_list)):
    edge_labels[(edge_list[i][0], edge_list[i][1])] = edge_list[i][2]
# エッジをまとめて追加する
G.add_weighted_edges_from(edge_list)
# ノードを描画する位置を決める
pos = nx.spring_layout(G)
# 描画
nx.draw_networkx(G, pos)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
# plt.show()

adj = nx.adjacency_matrix(G, nodelist = range(7))
print(nx.adjacency_matrix(G)[(4, 6)])