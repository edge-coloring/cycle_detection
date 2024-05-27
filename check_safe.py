# %%
from parse_prim import ReadPrimalFromFile, LabelEdges
import numpy as np
import sys
import itertools
import copy

PetersenOnlyMode = True

assert len(sys.argv) == 3
filePath = sys.argv[1]
contEdgeStr = sys.argv[2]

n, r, G = ReadPrimalFromFile(filePath)
edgeIndexes = LabelEdges(n, r, G)
e = len(edgeIndexes)
edges = [(u,v) for ((u,v),_) in edgeIndexes.items()]

A = [[] for _ in range(n)]

for (u,v), _ in edgeIndexes.items():
    A[u].append(v)
    A[v].append(u)

contEdges = list(map(int, contEdgeStr.split("+")))
# %%
def WF(dist, n):
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

dist = [[10000 for _ in range(n)] for _ in range(n)]
dist_hect = [[10000 for _ in range(n)] for _ in range(n)]
dist_orig = [[10000 for _ in range(n)] for _ in range(n)]
for i in range(n): 
    dist_hect[i][i] = 0
    dist_orig[i][i] = 0
for (u,v), e in edgeIndexes.items():
    cost = 0 if e in contEdges else 1
    cost_hect = 1 if e in contEdges else 100
    dist[u][v] = cost
    dist[v][u] = cost
    dist_hect[u][v] = cost_hect
    dist_hect[v][u] = cost_hect
    dist_orig[u][v] = 1
    dist_orig[v][u] = 1

WF(dist_hect, n)
WF(dist_orig, n)

# %%
def getComponentPair(path, vs, us):
    flag = [x in path for x in range(n)]
    one = []
    for i in vs:
        if not flag[i]:
            def get(v):
                flag[v] = True
                cnt = [v]
                for u in A[v]:
                    if not flag[u]:
                        cnt += get(u)
                return cnt
            one += get(i)
    two = []
    for i in us:
        if not flag[i]:
            def get(v):
                flag[v] = True
                cnt = [v]
                for u in A[v]:
                    if not flag[u]:
                        cnt += get(u)
                return cnt
            two += get(i)
    return [(one), (two)]

# %%
def getComponentsCurl(path):
    outer, inner = getComponentPair(path, range(r), range(r, n))
    return [(outer), (inner)]

# %%
def getComponentsCross(path, s, t):
    res1, res2 = getComponentPair(path, range(min(s,t), max(s,t)), map(lambda x: x % r, list(range(max(s,t), min(s,t)+r))))
    return [res1, res2]

def getComponentsCrossTwo(path1, path2, p1, q1, p2, q2):
    inc = list(range(min(p1, q1), max(p1, q1))) + list(range(min(p2, q2), max(p2, q2)))
    exc = list(set(range(r)) - set(inc))
    res1, res2 = getComponentPair(path1 + path2, exc, inc)
    return [res1, res2]

# %%
def dfs(thread, end, white, black, pool):
    if white == 0 and black == 0:
        if thread[-1] == end:
            pool.append(thread)
        return
    if white > 0:
        for next in A[thread[-1]]:
            if next not in thread and dist[next][thread[-1]] == 1:
                dfs(thread + [next], end, white - 1, black, pool)
    if black > 0:
        for next in A[thread[-1]]:
            if next not in thread and dist[next][thread[-1]] == 0:
                dfs(thread + [next], end, white, black - 1, pool)

def dfs2(thread, end, all, pool):
    if thread[-1] == end:
        pool.append(thread)
        return
    if all > 0:
        for next in A[thread[-1]]:
            if next not in thread:
                dfs2(thread + [next], end, all - 1, pool)

innerExists = [[False for _ in range(r)] for _ in range(r)]
for p in range(r):
    for q in range(p+1, r):
        d = dist_orig[p][q]
        pool = []
        dfs2([p], q, d, pool)
        for path in pool:
            if max(path) >= r:
                innerExists[p][q] = True
                break

# %%
# calculate after degree
def getContSet(v, pres):
    for u in A[v]:
        if dist[u][v] > 0: continue
        if u in pres: continue
        pres.add(u)
        getContSet(u, pres)
    return pres
contSet = [getContSet(i, {i}) for i in range(n)]

# %%
def getInnerCycles(edges: list[(int, int)]):
    cycles = []
    edgesNum = len(edges)
    assert(1 <= edgesNum <= 3)
    for bit in range([1, 2, 8][edgesNum-1]):
        flipEdges = copy.deepcopy(edges)
        for j in range(edgesNum):
            if (1 << j) & bit:
                u, v = flipEdges[edgesNum - 1 - j]
                flipEdges[edgesNum - 1 - j] = (v, u)
        whites = []
        blacks = []
        for j in range(edgesNum):
            dh = dist_hect[flipEdges[j][1]][flipEdges[(j + 1) % edgesNum][0]]
            whites.append(dh // 100)
            blacks.append(dh % 100)
        if sum(whites) == 0:
            pools = [[] for _ in range(edgesNum)]
            for j in range(edgesNum):
                vnow = flipEdges[j][1]
                unext = flipEdges[(j + 1) % edgesNum][0]
                dfs([vnow], unext, whites[j], blacks[j], pools[j])
                assert len(pools[j]) > 0
            for pickedPaths in itertools.product(*pools):
                path = sum(pickedPaths, [])
                comps = getComponentsCurl(path)
                if min(map(len, comps)) == 0: continue
                if min(comps) == 0: continue
                if len(path) < 5: continue
                cycles.append((path, comps, edgesNum))
    return cycles

# %%
cycles = []
for a in range(e):
    if a in contEdges:
        continue
    cycles += getInnerCycles([edges[a]])
if len(cycles) > 0:
    print("Dangerous (inner bridge)")
    print(f"{filePath}", file=sys.stderr)
                

# %%
# skip = 1
hasDangerousCross = False
outwardComps = set()
def isLimitableCut(comps, originalCut):
    global hasDangerousCross
    if originalCut <= 5:
        return True
    if originalCut >= 8:
        hasDangerousCross = True
        return False
    limit = 3 if originalCut == 6 else 4
    if min(map(len, comps)) > limit:
        return True 
    if max(map(len, comps)) <= limit:
        hasDangerousCross = True
        return False
    else:
        for comp in comps:
            if len(comp) <= 3:
                for v in comp:
                    outwardComps.add(v)
        return False
    assert False

# %%\
for p in range(r):
    for q in range(p+1, r):
        d = dist_orig[p][q]
        pool = []
        cc = max(1, 5 - d)
        for c in range(cc, 3+1):
            e = 3 - c
            white = dist_hect[p][q] // 100
            black = dist_hect[p][q] % 100
            if white <= e:
                # check for shorter cut 
                pool = []
                dfs2([p], q, white + black, pool)
                originalCut = c + white + black
                shorterCutAvailable = False
                for path in pool:
                    comps = getComponentsCross(path, path[0], path[-1])
                    assert originalCut >= 5
                    newCut = len(path) + c - 1
                    assert newCut <= originalCut
                    if newCut >= 8:
                        continue
                    limit = 1 if newCut == 5 else 3 if newCut == 6 else 4
                    if min(map(len, comps)) > limit:
                        shorterCutAvailable = True
                if shorterCutAvailable:
                    continue
                # check for contractable cut
                pool = []
                dfs([p], q, white, black, pool)
                for path in pool:
                    originalCut = c + white + black
                    # cut removal
                    comps = getComponentsCross(path, path[0], path[-1])
                    limitable = isLimitableCut(comps, originalCut)
                    if limitable:
                        continue
                    print(f"{c + white + black} -> {c + white} : {c} {path} ({comps})")
# %%
# skip = 2
hasTwo = False
for p1 in range(r):
    for q1 in range(p1+1, r):
        d1 = dist_orig[p1][q1]
        c1 = max(1, (6 if innerExists[p1][q1] else 5) - d1)
        for p2 in range(r):
            for q2 in range(p2+1, r):
                rings = sorted([p1, q1, p2, q2])
                if (p1 == q1 and p2 == q2):
                    continue
                if not (p1 in [rings[0], rings[2]] and p2 in [rings[0], rings[2]]):
                    if not (q1 in [rings[0], rings[2]] and q2 in [rings[0], rings[2]]):
                        continue
                d2 = dist_orig[p2][q2]
                c2 = max(1, (6 if innerExists[p2][q2] else 5) - d2)
                esum = 3 - c1 - c2
                white1 = dist_hect[q1][p2] // 100
                black1 = dist_hect[q1][p2] % 100
                white2 = dist_hect[q2][p1] // 100
                black2 = dist_hect[q2][p1] % 100
                if white1 + white2 <= esum and c1 + dist_orig[q1][p2] + c2 + dist_orig[q2][p1] >= 6:
                    pool1 = []
                    dfs([q1], p2, white1, black1, pool1)
                    pool2 = []
                    dfs([q2], p1, white2, black2, pool2)
                    for path1 in pool1:
                        for path2 in pool2:
                            comps = getComponentsCrossTwo(path1, path2, p1, q1, p2, q2)
                            originalCut = c1 + white1 + black1 + c2 + white2 + black2
                            # cut removal
                            if originalCut == 6:
                                if min(map(len, comps)) > 3:
                                    continue 
                                if max(map(len, comps)) <= 3:
                                    hasDangerousCross = True
                                else:
                                    for comp in comps:
                                        if len(comp) <= 3:
                                            for v in comp:
                                                outwardComps.add(v)
                            if originalCut == 7:
                                if min(map(len, comps)) > 4:
                                    continue
                                if max(map(len, comps)) <= 4:
                                    hasDangerousCross = True
                                else:
                                    for comp in comps:
                                        if len(comp) <= 4:
                                            for v in comp:
                                                outwardComps.add(v)
                            if originalCut >= 8:
                                hasDangerousCross = True
                            print(f"{c1 + white1 + black1 + c2 + white2 + black2} -> {c1 + white1 + c2 + white2} : {c1} {path1} {c2} {path2} ({comps})")
                            hasTwo = True

# %%
class Graph:
    verts: list[tuple]
    edges: set[tuple[tuple, tuple]]
    isOuter: dict[tuple, bool]
    what: dict[int, tuple]
    def __init__(self, n, r, edges, contSet):
        self.verts = []
        for i in range(n):
            self.verts.append(tuple(sorted(list(contSet[i]))))
        self.verts = list(set(self.verts))
        self.checkNoOverlaps()
        self.what = {}
        for _, vs in enumerate(self.verts):
            for v in vs:
                self.what[v] = vs
        self.edges = set()
        for u,v in edges:
            self.edges.add((self.what[u], self.what[v]))
            self.edges.add((self.what[v], self.what[u]))
        self.isOuter = {}
        for vs in self.verts:
            for v in vs:
                if v < r:
                    self.isOuter[vs] = True
                    break
            else:
                self.isOuter[vs] = False
    def checkNoOverlaps(self):
        for b in range(len(self.verts)):
            for a in range(b):
                avs = self.verts[a]
                bvs = self.verts[b]
                if len(set(avs).intersection(set(bvs))) > 0:
                    assert False
    def deleteVS(self, vs):
        self.verts.remove(vs)
        for us in self.verts:
            if (us, vs) in self.edges:
                self.edges.remove((us, vs))
                self.edges.remove((vs, us))
        self.isOuter.pop(vs)
        for v in vs:
            self.what.pop(v)
        self.checkNoOverlaps()
    def tryRemovingTheseCuts(self, cutPoints):
        flag = {vs: vs in cutPoints for vs in self.verts}
        def dfs(vs):
            flag[vs] = True
            res = [vs]
            for us in self.verts:
                if not flag[us] and (us, vs) in self.edges:
                    res += dfs(us)
            return res
        for vs in self.verts:
            if not flag[vs]:
                cons = dfs(vs)
                if not any(map(lambda vs: self.isOuter[vs], cons)):
                    # print(f"{len(cutPoints)} cut: {cutPoints} ({cons})", file=sys.stderr)
                    break
        else:
            return False
        for dvs in cons:
            self.deleteVS(dvs)
        return True
    def remove2or3cuts(self):
        print(f"Remaining: {self.verts}")
        for a in range(len(self.verts)):
            for b in range(a):
                if self.tryRemovingTheseCuts([self.verts[b], self.verts[a]]):
                    print(f"Cut: {b,a} th")
                    return True
        for a in range(len(self.verts)):
            for b in range(a):
                for c in range(b):
                    if self.tryRemovingTheseCuts([self.verts[c], self.verts[b], self.verts[a]]):
                        print(f"Cut: {c,b,a} th")
                        return True
        print("No cut found")
        return False
    def removeVSet(self, vset):
        vertsOld = (self.verts).copy()
        for vs in vertsOld:
            for v in vs:
                if v not in vset:
                    break
            else:
                self.deleteVS(vs)
    def checkGraphSize(self):
        rings = 0
        outers = 0
        inners = 0
        for vs in self.verts:
            if len(vs) == 1 and vs[0] < r:
                rings += 1
            elif self.isOuter[vs]:
                outers += 1
            else:
                inners += 1
        print(f"rings: {rings}, outers: {outers}, inners: {inners}")
        return rings * 0.5 + inners
    def checkDegree(self, vs):
        deg = 0
        for us in self.verts:
            if (us, vs) in self.edges:
                if us != vs:
                    deg += 1
        return deg
    def checkDegreeAll(self):
        dangerous = True
        for vs in self.verts:
            deg = self.checkDegree(vs)
            print(f"[{'outer' if self.isOuter[vs] else 'inner'}] delta({vs}) = {deg}", end=": ")
            if PetersenOnlyMode:
                if self.isOuter[vs]:
                    if deg >= 6:
                        dangerous = False
                        print("OK!")
                    else:
                        print("...")
                else:
                    if deg != 5:
                        dangerous = False
                        print("OK!")
                    else:
                        print("...")
            else:
                if not self.isOuter[vs] and deg == 4:
                    dangerous = False
                    print("OK! (four)")
                else:
                    print("...")
        if not PetersenOnlyMode:
            # check tri 5s
            print("Check tris...")
            for i1, vs1 in enumerate(self.verts):
                deg1 = self.checkDegree(vs1)
                o1 = self.isOuter[vs1]
                for i2, vs2 in enumerate(self.verts):
                    deg2 = self.checkDegree(vs2)
                    o2 = self.isOuter[vs2]
                    for i3, vs3 in enumerate(self.verts):
                        deg3 = self.checkDegree(vs3)
                        o3 = self.isOuter[vs3]
                        if i1 < i2 < i3:
                            if (vs1, vs2) in self.edges and (vs2, vs3) in self.edges and (vs3, vs1) in self.edges:
                                print(f'{vs1}, {vs2}, {vs3} = {deg1}{"+" if o1 else ""}, {deg2}{"+" if o2 else ""}, {deg3}{"+" if o3 else ""}')
                                if deg1 >= 6 and deg2 >= 6 and deg3 >= 6:
                                    print(f"OK! (tri)")
                                    dangerous = False
        return dangerous

                
# %%
if hasDangerousCross:
    print("Dangerous (has cross)")
    print(f"{filePath}", file=sys.stderr)
    exit(0)
graph = Graph(n, r, edges, contSet)
print(f"Removing: {outwardComps}")
graph.removeVSet(outwardComps)
while graph.remove2or3cuts(): pass
graphSizeMax = 6 if PetersenOnlyMode else 18
if (graphSize := graph.checkGraphSize()) > graphSizeMax:
    print(f"Lower bound of graph size = {graphSize} > {graphSizeMax}: Safe")
    exit(0)
else:
    print(f"Lower bound of graph size = {graphSize} <= {graphSizeMax}...")
if not graph.checkDegreeAll():
    print(f"Degree mismatch: Safe")
    exit(0)
else:
    print(f"Degree matches...")

print("Dangerous (checked)")
print(f"{filePath}", file=sys.stderr)
