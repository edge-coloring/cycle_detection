def ReadPrimalFromFile(filePath: str):
    with open(filePath, "r") as f:
        f.readline()
        n, r = map(int, f.readline().split())
        G = [[] for _ in range(n-r)]
        for i in range(n-r):
            v, m, *us = map(int, f.readline().split())
            G[v-r-1] = list(map(lambda x: x-1,us))
        return n, r, G
    assert False

def LabelEdges(n:int, r: int, G: "list[list[int]]"):
    edges = set()
    for i in range(r):
        edges.add((i, (i + 1) % r))
        edges.add(((i + 1) % r, i))
    for i in range(n - r):
        for j in G[i]:
            edges.add((i + r, j))
            edges.add((j, i + r))
    def is3Cycle(i: int, j: int, k: int):
        return (i, j) in edges and (j, k) in edges and (k, i) in edges
    triangles = set()
    for i in range(n):
        for j in range(i):
            for k in range(j):
                if is3Cycle(k, j, i):
                    triangles.add((k, j, i))
    triangles = sorted(list(triangles))
    edgeIndexes = {}
    def addEdge(x, y):
        if x > y: x, y = y, x
        if (x, y) not in edgeIndexes:
            edgeIndexes[(x, y)] = len(edgeIndexes)
    for i in range(r):
        addEdge(i, (i + 1) % r)
    for a,b,c in triangles:
        addEdge(a, b)
        addEdge(b, c)
        addEdge(c, a)
    return edgeIndexes