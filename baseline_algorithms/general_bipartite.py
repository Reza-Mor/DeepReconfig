import networkx as nx
import random
from networkx.algorithms.bipartite.generators import random_graph
from networkx.algorithms.bipartite import sets, maximum_matching, to_vertex_cover
from networkx.algorithms.components import condensation
from networkx.algorithms.dag import topological_sort
from copy import deepcopy

DEBUG = False


def vertex_cover_from_matching(B, R, G, M):

    def alt_path_augment(M, G, u):

        augment = set([])

        match_edges = [(vertex, M[vertex]) for vertex in M.keys()]

        color = {}

        for vertex in G.nodes:
            color[vertex] = 0

        color[u] = 1

        queue = [u]

        while len(queue) > 0:
            vertex = queue.pop()

            if color[vertex] == 2 and vertex in M.keys():
                if color[M[vertex]] == 0:
                    color[M[vertex]] = 1
                    queue.append(M[vertex])
                    augment.add(M[vertex])

            for v in G.neighbors(vertex):

                if color[vertex] == 1 and color[v] == 0 and (vertex, v) not in match_edges:
                    color[v] = 2
                    queue.append(v)
                    augment.add(v)

#                if color[vertex] == 2 and color[v] == 0 and (vertex, v) in match_edges:
#                    color[v] = 1
#                    queue.append(v)
#                    augment.add(v)

        return augment

    Z = set([])
    Augment = set([])

    for u in G.nodes:
        if u in B and u not in M.keys():
            Z.add(u)

    for u in Z:
        Augment = Augment.union(alt_path_augment(M, G, u))

    Z = Z.union(Augment)

    return (B - Z).union(R.intersection(Z))


def random_bipartite(n, degree, seed=None):

    if seed:
        random.seed(seed)

    B = set(list(range(n)))
    R = set(list(range(n, 2*n, 1)))

    G = nx.Graph()

    for u in range(2*n):
        G.add_node(u)

    for u in B:
        for v in R:
            if random.random() < float(degree)/float(n):
                G.add_edge(u, v)

    return B, R, G


def coarseDulmageMendelsohn(M, G):

    def even_alt_path_augment(M, G, u):

        augment = set([])

        match_edges = [(vertex, M[vertex]) for vertex in M.keys()]

        color = {}

        for vertex in G.nodes:
            color[vertex] = 0

        color[u] = 1

        queue = [u]

        while len(queue) > 0:
            vertex = queue.pop()

            if color[vertex] == 2 and vertex in M.keys():
                if color[M[vertex]] == 0:
                    color[M[vertex]] = 1
                    queue.append(M[vertex])
                    augment.add(M[vertex])

            for v in G.neighbors(vertex):

                if color[vertex] == 1 and color[v] == 0 and (vertex, v) not in match_edges:
                    color[v] = 2
                    queue.append(v)

#                if color[vertex] == 2 and color[v] == 0 and (vertex, v) in match_edges:
#                    color[v] = 1
#                    queue.append(v)
#                    augment.add(v)

        return augment

    D = set([])
    Augment = set([])

    for u in G.nodes:
        if u not in M.keys():
            D.add(u)

    for u in D:
        Augment = Augment.union(even_alt_path_augment(M, G, u))

    D = D.union(Augment)

    A = set([])

    for u in D:
        for v in G.neighbors(u):
            A.add(v)

    C = set([])

    for u in G.nodes:
        if u not in D and u not in A:
            C.add(u)

    return D, A, C


def mixedMIS(B, R, G):

    M = maximum_matching(G, top_nodes=B)

    if DEBUG:
        print("maximum matching: ", M)

    K = vertex_cover_from_matching(B, R, G, M)

    I = set(G) - K

    if len(I.intersection(B)) > 0 and len(R.intersection(I)) > 0:
        return I

    D, A, C = coarseDulmageMendelsohn(M, G)

    if DEBUG:
        print("D: ", D, " A: ", A, "C: ", C)

    if len(B) > len(R):
        if len(R - A) > 0:
            r = next(iter(R-A))

            Gp = G.copy()

            new_B = B.copy()
            new_R = R.copy()

            for v in G.neighbors(r):
                Gp.remove_node(v)
                new_B.remove(v)
            Gp.remove_node(r)
            new_R.remove(r)

            Mp = maximum_matching(Gp, top_nodes=new_B)
            Kp = vertex_cover_from_matching(new_B, new_R, Gp, Mp)
            Ip = set(Gp) - Kp

            Ip.add(r)

            return Ip

        else:
            return None

    if len(B) == len(R):

        H = nx.DiGraph()

        for u, v in M.items():

            if u in B:
                H.add_node((u, v))
            if v in B:
                H.add_node((v, u))

        for t1 in H.nodes:
            for t2 in H.nodes:
                if t1 != t2:
                    if G.has_edge(t1[0], t2[1]):
                        H.add_edge(t1, t2)

        if DEBUG:
            print("H ", H.adj)

        Cond = condensation(H)

        if Cond.number_of_nodes() <= 1:
            return None
        else:
            sorted_vertices = topological_sort(Cond)

            mMIS = set([])

            for k, V in enumerate(sorted_vertices):
                if k == 0:
                    for u in Cond.nodes[V]['members']:
                        mMIS.add(u[1])
                else:
                    for u in Cond.nodes[V]['members']:
                        mMIS.add(u[0])

            return mMIS


def realize(B, R, G, k, depth=0):
    """
    Main function for solving **bipartite independent set reconfiguration**, computing whether it is possible to reconfigure B into R while always having at least :math:`|B|-k` vertices. 

    :param B: Set of vertices of G, the **left** side of G.
    :type B: Python **set**

    :param R: Set of vertices of G, the **right** side of G.
    :type R: Python **set**

    :param G: Input bipartite graph.
    :type G: nx.Graph (networkx graph)

    :param k: barrier value. The purspose of the function is to decide whether the input graph has a reconfiguration pathway from B to R with barrier k or not.
    :type k: int

    :return: **P** : either **None** or a **list** of the vertices of G according to a feasible reconfiguration pathway.
    :rtype: **list** or **None** 
    """

    if DEBUG:
        print("B,R,G,k", B, R, G, k)

    if DEBUG:
        for _ in range(depth):
            print("    ", end="")

    if k < 0:
        return None

    if len(B) == 0:
        return list(R)
    if len(R) == 0:
        if len(B) > k:
            return None
        else:
            return list(B)

    if len(B) <= k:
        return list(B) + list(R)

    if len(B) < len(R):
        k += len(R) - len(B)  # update k
        res = realize(R, B, G, k, depth=depth)

        if res:
            res.reverse()
            return res
        else:
            return None

    I = mixedMIS(B, R, G)

    if I:

        S = (B - I).union(R.intersection(I))
        delta_S = len(S.intersection(B)) - len(S.intersection(R))

        first_part = realize(B.intersection(
            S), R.intersection(S), G.subgraph(S), k)
        if first_part:
            second_part = realize(B.intersection(set(G)-S),
                                  R.intersection(set(G)-S),
                                  G.subgraph(set(G)-S), k-delta_S)

            if second_part:
                return first_part + second_part

    else:
        for r in R:

            ngbh_r = set([])

            for b in G.neighbors(r):
                ngbh_r.add(b)

            if len(ngbh_r) <= k:

                Gp = G.copy()
                Gp.remove_node(r)

                for b in ngbh_r:
                    Gp.remove_node(b)

                res = realize(B-ngbh_r, R-set([r]), Gp, k-len(ngbh_r)+1)
                if res:
                    return list(ngbh_r)+[r] + res

        return None


if __name__ == "__main__":

    #    G = nx.Graph()
    #
    #    for u in range(6):
    #        G.add_node(u)
    #
    #    for u in range(3):
    #        for v in range(3,6,1):
    #            G.add_edge(u,v)
    #
    #    B, R = sets(G)
    #
    #    print("1: B, R", B, R)
    #    print("mixed MIS 1", mixedMIS(B,R,G))
    #
    #    G = nx.Graph()
    #
    #    for u in range(6):
    #        G.add_node(u)
    #
    #    G.add_edge(0, 3)
    #    G.add_edge(0, 4)
    #    G.add_edge(0, 5)
    #    G.add_edge(1, 3)
    #    G.add_edge(1, 4)
    #    G.add_edge(1, 5)
    #    G.add_edge(2, 4)
    #
    #    B, R = sets(G)
    #
    #    print("2: B, R", B, R)
    #    print("mixed MIS 2", mixedMIS(B,R,G))
    #
    #    G = nx.Graph()
    #
    #    for u in range(7):
    #        G.add_node(u)
    #
    #    G.add_edge(0,4)
    #    G.add_edge(0,5)
    #    G.add_edge(0,6)
    #    G.add_edge(1,5)
    #    G.add_edge(1,6)
    #    G.add_edge(2,5)
    #    G.add_edge(2,6)
    #    G.add_edge(3,5)
    #    G.add_edge(3,6)
    #
    #    B, R = sets(G)
    #
    #    print("3: B, R", B, R)
    #    print("mixed MIS 3", mixedMIS(B,R,G))
    #
    #    G = nx.Graph()
    #
    #    for u in range(7):
    #        G.add_node(u)
    #
    #    for u in range(4):
    #        for v in range(4, 7, 1):
    #            G.add_edge(u,v)
    #
    #    B, R = sets(G)
    #
    #    print("4: B, R", B, R)
    #    print("mixed MIS 4", mixedMIS(B,R,G))
    #
    #    G = nx.Graph()
    #
    #    for u in range(7):
    #        G.add_node(u)
    #
    #    G.add_edge(0,4)
    #    G.add_edge(0,5)
    #    G.add_edge(0,6)
    #    G.add_edge(1,5)
    #    G.add_edge(1,6)
    #    G.add_edge(2,5)
    #    G.add_edge(2,6)
    #    G.add_edge(3,5)
    #    G.add_edge(3,6)
    #
    #    B, R = sets(G)
    #
    #    print("5: B, R", B, R)
    #    print("mixed MIS 5", mixedMIS(B,R,G))
    #
    #    G = nx.Graph()
    #
    #    for u in range(6):
    #        G.add_node(u)
    #
    #    G.add_edge(0,4)
    #    G.add_edge(1,3)
    #    G.add_edge(1,4)
    #    G.add_edge(1,5)
    #    G.add_edge(2,4)
    #    G.add_edge(2,5)
    #
    #    B, R = sets(G)
    #
    #    print("6: B, R", B, R)
    #    print("mixed MIS 6", mixedMIS(B,R,G))
    #
    #    G = nx.Graph()
    #
    #    for u in range(6):
    #        G.add_node(u)
    #
    #    G.add_edge(0,4)
    #    G.add_edge(1,3)
    #    G.add_edge(1,4)
    #    G.add_edge(1,5)
    #    G.add_edge(2,4)
    #
    #    B, R = sets(G)
    #
    #    print("7: B, R", B, R)
    #    print("mixed MIS 7", mixedMIS(B,R,G))

    # Set random seed for sake of reproducibility

    G = nx.Graph()

    for u in range(6):
        G.add_node(u)

    G.add_edge(0, 4)
    G.add_edge(1, 3)
    G.add_edge(1, 4)
    G.add_edge(1, 5)
    G.add_edge(2, 4)
    G.add_edge(2, 5)

    B, R = sets(G)

    for u in G.nodes:
        print(u, ": ", list(G.neighbors(u)))

    print("7: B, R", B, R)

    k = 0
    while True:
        # try with k
        print("k: ", k)
        P = realize(B, R, G, k)
        if P:
            break
        k += 1

    print("1: found k=", k)
    print("1: found P=", P)

    G = nx.Graph()

    for u in range(7):
        G.add_node(u)

    G.add_edge(0, 4)
    G.add_edge(1, 3)
    G.add_edge(1, 4)
    G.add_edge(1, 5)
    G.add_edge(2, 4)
    G.add_edge(2, 5)
    G.add_edge(2, 6)
    G.add_edge(1, 6)

    B, R = sets(G)

    for u in G.nodes:
        print(u, ": ", list(G.neighbors(u)))

    print("7: B, R", B, R)

    k = 0
    while True:
        # try with k
        print("k: ", k)
        P = realize(B, R, G, k)
        if P:
            break
        k += 1

    print("2: found k=", k)
    print("2: found P=", P)

    G = nx.Graph()

    for u in range(7):
        G.add_node(u)

    for u in range(3):
        for v in range(3, 7, 1):
            G.add_edge(u, v)

    B, R = sets(G)

    for u in G.nodes:
        print(u, ": ", list(G.neighbors(u)))

    print("7: B, R", B, R)

    k = 0
    while True:
        # try with k
        print("k: ", k)
        P = realize(B, R, G, k)
        if P:
            break
        k += 1

    print("3: found k=", k)
    print("3: found P=", P)

    G = nx.Graph()

    for u in range(6):
        G.add_node(u)

    for u in range(3):
        for v in range(3, 6, 1):
            G.add_edge(u, v)

    B, R = sets(G)

    for u in G.nodes:
        print(u, ": ", list(G.neighbors(u)))

    print("7: B, R", B, R)

    k = 0
    while True:
        # try with k
        print("k: ", k)
        P = realize(B, R, G, k)
        if P:
            break
        k += 1

    print("4: found k=", k)
    print("4: found P=", P)

    print("BUGGY EXAMPLE: ")

    B, R, G = random_bipartite(8, 4, seed=3)

    print("B, R: ", B, R)
    for u in G.nodes:
        print(u, ": ", list(G.neighbors(u)))

    k = 0
    while True:
        # try with k
        print("k: ", k)
        P = realize(B, R, G, k)
        if P:
            break
        k += 1

    print("found k=", k)
    print("found P=", P)

    B, R, G = random_bipartite(5, 3, seed=4)

    print("B, R: ", B, R)
    for u in G.nodes:
        print(u, ": ", list(G.neighbors(u)))

    k = 0
    while True:
        # try with k
        print("k: ", k)
        P = realize(B, R, G, k)
        if P:
            break
        k += 1

    print("found k=", k)
    print("found P=", P)
