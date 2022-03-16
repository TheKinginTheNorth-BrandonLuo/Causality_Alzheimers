import itertools
from itertools import combinations, chain
from scipy.stats import norm, pearsonr
import math
import networkx as nx

def subset(iterable):
    # get subset
    # subset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    xs = list(iterable)
    # return iterator, not list
    return chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1))


def skeleton(suffStat, indepTest, alpha, labels, method,
             fixedGaps, fixedEdges,
             NAdelete, m_max, numCores, verbose):

    sepset = [[[] for i in range(len(labels))] for i in range(len(labels))]

    # undirected graph
    G = [[True for i in range(len(labels))] for i in range(len(labels))]

    for i in range(len(labels)):
        # don't need to test i--i
        G[i][i] = False

    # indicator: done flag
    done = False

    ord = 0
    n_edgetests = {0: 0}
    while done != True and any(G) and ord <= m_max:
        ord1 = ord + 1
        n_edgetests[ord1] = 0

        done = True

        # adjacent points
        ind = []
        for i in range(len(G)):
            for j in range(len(G[i])):
                if G[i][j] == True:
                    ind.append((i, j))

        G1 = G.copy()

        for x, y in ind:
            if G[x][y] == True:
                neighborsBool = [row[x] for row in G1]
                neighborsBool[y] = False

                # adj(C,x)/{y}
                neighbors = [i for i in range(len(neighborsBool)) if neighborsBool[i] == True]

                if len(neighbors) >= ord:

                    # |adj(C, x) / {y}|>ord
                    if len(neighbors) > ord:
                        done = False

                    # |adj(C, x) / {y}|=ord
                    for neighbors_S in set(itertools.combinations(neighbors, ord)):

                        n_edgetests[ord1] = n_edgetests[ord1] + 1

                        # whether x,y d-seperation by neighbors_S
                        # test conditional independence and return to p-value
                        pval = indepTest(suffStat, x, y, list(neighbors_S))

                    

                        # conditional indepence
                        if pval >= alpha:
                            G[x][y] = G[y][x] = False

                            # add neighbor_S to seperate set
                            sepset[x][y] = list(neighbors_S)
                            break

        ord += 1

    return {'sk': np.array(G), 'sepset': sepset}

def extend_cpdag(graph):
    # Rule 1: change a -> b - c  to a -> b -> c
    def rule1(pdag, solve_conf=False, unfVect=None):
        search_pdag = pdag.copy()
        ind = []
        for i in range(len(pdag)):
            for j in range(len(pdag)):
                if pdag[i][j] == 1 and pdag[j][i] == 0:
                    ind.append((i, j))

        #
        for a, b in sorted(ind, key=lambda x:(x[1],x[0])):
            isC = []
            for i in range(len(search_pdag)):
                if (search_pdag[b][i] == 1 and search_pdag[i][b] == 1) and (search_pdag[a][i] == 0 and search_pdag[i][a] == 0):
                    isC.append(i)
            if len(isC) > 0:
                for c in isC:
                    if 'unfTriples' in graph.keys() and ((a, b, c) in graph['unfTriples'] or (c, b, a) in graph['unfTriples']):
                        # if unfaithful, skip
                        continue
                    if pdag[b][c] == 1 and pdag[c][b] == 1:
                        pdag[b][c] = 1
                        pdag[c][b] = 0
                    elif pdag[b][c] == 0 and pdag[c][b] == 1:
                        pdag[b][c] = pdag[c][b] = 2
        return pdag

    # Rule 2: if chain a -> c -> b exists，change a - b to a -> b
    def rule2(pdag, solve_conf=False):
        search_pdag = pdag.copy()
        ind = []
        for i in range(len(pdag)):
            for j in range(len(pdag)):
                if pdag[i][j] == 1 and pdag[j][i] == 1:
                    ind.append((i, j))

        #
        for a, b in sorted(ind, key=lambda x:(x[1],x[0])):
            isC = []
            for i in range(len(search_pdag)):
                if (search_pdag[a][i] == 1 and search_pdag[i][a] == 0) and (search_pdag[i][b] == 1 and search_pdag[b][i] == 0):
                    isC.append(i)
            if len(isC) > 0:
                if pdag[a][b] == 1 and pdag[b][a] == 1:
                    pdag[a][b] = 1
                    pdag[b][a] = 0
                elif pdag[a][b] == 0 and pdag[b][a] == 1:
                    pdag[a][b] = pdag[b][a] = 2
        return pdag

    # Rule 3: if a - b, a - c1 exists, a - c2, c1 -> b, c2 -> b，and c1, c2 not adjacent，change a - b to a -> b
    def rule3(pdag, solve_conf=False, unfVect=None):
        search_pdag = pdag.copy()
        ind = []
        for i in range(len(pdag)):
            for j in range(len(pdag)):
                if pdag[i][j] == 1 and pdag[j][i] == 1:
                    ind.append((i, j))

        #
        for a, b in sorted(ind, key=lambda x:(x[1],x[0])):
            isC = []
            for i in range(len(search_pdag)):
                if (search_pdag[a][i] == 1 and search_pdag[i][a] == 1) and (search_pdag[i][b] == 1 and search_pdag[b][i] == 0):
                    isC.append(i)
            if len(isC) >= 2:
                for c1, c2 in combinations(isC, 2):
                    if search_pdag[c1][c2] == 0 and search_pdag[c2][c1] == 0:
                        # unfaithful
                        if 'unfTriples' in graph.keys() and ((c1, a, c2) in graph['unfTriples'] or (c2, a, c1) in graph['unfTriples']):
                            continue
                        if search_pdag[a][b] == 1 and search_pdag[b][a] == 1:
                            pdag[a][b] = 1
                            pdag[b][a] = 0
                            break
                        elif search_pdag[a][b] == 0 and search_pdag[b][a] == 1:
                            pdag[a][b] = pdag[b][a] = 2
                            break
        return pdag

    # Rule 4: 如果存在链 i - k -> l 和 k -> l -> j 且 k 和 j 不相邻，把 i - j 改为 i -> j
    # 但论文认为 rule4 在这种情况下并不必要

    pdag = [[0 if graph['sk'][i][j] == False else 1 for i in range(len(graph['sk']))] for j in range(len(graph['sk']))]

    ind = []
    for i in range(len(pdag)):
        for j in range(len(pdag[i])):
            if pdag[i][j] == 1:
                ind.append((i, j))

    #
    # 把 x - y - z 变为 x -> y <- z
    for x, y in sorted(ind, key=lambda x:(x[1],x[0])):
        allZ = []
        for z in range(len(pdag)):
            if graph['sk'][y][z] == True and z != x:
                allZ.append(z)

        for z in allZ:
            if graph['sk'][x][z] == False and graph['sepset'][x][z] != None and graph['sepset'][z][x] != None and not (
                    y in graph['sepset'][x][z] or y in graph['sepset'][z][x]):
                pdag[x][y] = pdag[z][y] = 1
                pdag[y][x] = pdag[y][z] = 0

    # 应用rule1-3
    pdag = rule1(pdag)
    pdag = rule2(pdag)
    pdag = rule3(pdag)

    return np.array(pdag)


def pc(suffStat, alpha, labels, indepTest, p='Use labels',
       fixedGaps=None, fixedEdges=None, NAdelete=True, m_max=float("inf"),
       u2pd=("relaxed", "rand", "retry"),
       skel_method=("stable", "original", "stable.fast"),
       conservative=False, maj_rule=False, solve_confl=False,
       numCores=1, verbose=False):

    # 骨架
    graphDict = skeleton(suffStat, indepTest, alpha, labels=labels, method=skel_method,
                         fixedGaps=fixedGaps, fixedEdges=fixedEdges,
                         NAdelete=NAdelete, m_max=m_max, numCores=numCores, verbose=verbose)

    # 扩展为CPDAG
    return extend_cpdag(graphDict)

# 条件独立性检验
def gaussCItest(suffstat, x, y, S):
    C = suffstat["C"]
    n = suffstat["n"]

    cut_at = 0.9999999

    # 偏相关系数
    # S中没有点
    if len(S) == 0:
        r = C[x, y]

    # S中只有一个点 一阶偏相关系数
    elif len(S) == 1:
        r = (C[x, y] - C[x, S] * C[y, S]) / math.sqrt((1 - math.pow(C[y, S], 2)) * (1 - math.pow(C[x, S], 2)))

    # 其实我没太明白这里是怎么求的，但R语言的pcalg包就是这样写的
    else:
        m = C[np.ix_([x]+[y]+S, [x]+[y]+S)]
        PM = np.linalg.pinv(m)
        r = -1 * PM[0, 1] / math.sqrt(abs(PM[0, 0] * PM[1, 1]))

    r = min(cut_at, max(-1*cut_at, r))
    # Fisher’s z-transform
    res = math.sqrt(n - len(S) - 3) * .5 * math.log1p((2 * r) / (1 - r))
    # Φ^{-1}(1-α/2)
    return 2 * (1 - norm.cdf(abs(res)))

def Dfs(graph, k, path, vis):
    flag = False
    for i in range(len(graph)):
        if (graph[i][k] == 1) and (vis[i] != True) :
            
            flag =True
            vis[i] = True
            path.append(i)
            Dfs(graph, i, path, vis)
            path.pop()
            vis[i] = False

    if flag == False:
        print(path)

def Draw(graph, labels):
    # 创建空有向图
    G = nx.DiGraph()

    for i in range(len(graph)):
        G.add_node(labels[i])
        for j in range(len(graph[i])):
            if graph[i][j] == 1:
                G.add_edges_from([(labels[i], labels[j])])

    nx.draw(G, with_labels=True)
    plt.savefig("result.png")
    plt.show()


if __name__ == '__main__':
    # read out dataset
    data = pd.read_csv("dt.csv")
    # create label list
    labels = [
        "PRP_NOUN_Ratio",
        "NOUN_Ratio",
        "PRP_Ratio",
        "H_S",
        "BM",
        "TTR",
        "SICH",
        "A_R_I",
        "F_K_R"
    ]
    #labels = ["PRP_NOUN_Ratio","word_freq_rate","verb_freq_rate","PRON_rate","ADV_rate","H_S","BM","SICH","A_R_I"]

    row_count = sum(1 for row in data)
    # inplement to pc function
    p = pc(suffStat={"C": data.corr().values, "n": data.values.shape[0]}, alpha=.05, labels=[str(i) for i in range(row_count)], indepTest=gaussCItest)
    # output causal matrix
    print(p)

    # Dfs causal relation chain
    start = 2 # start anomaly node
    vis = [0 for i in range(row_count)]
    vis[start] = True
    path = []
    path.append(start)
    Dfs(p, start, path, vis)

    # draw graph
    Draw(p, labels)
