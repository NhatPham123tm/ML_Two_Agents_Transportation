# BFS shortest path on a grid with obstacles (ignores other agent)
from typing import List, Tuple, Set
from collections import deque

Coord = Tuple[int,int]

def shortest_path(start: Coord, goal: Coord, H:int, W:int, obstacles:Set[Coord]) -> List[Coord]:
    if start == goal:
        return [start]
    def inside(r,c): return 0 <= r < H and 0 <= c < W and (r,c) not in obstacles
    q = deque([start])
    parent = {start: None}
    while q:
        r,c = q.popleft()
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            v = (nr,nc)
            if not inside(nr,nc) or v in parent:
                continue
            parent[v] = (r,c)
            if v == goal:
                # reconstruct
                path = [v]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                path.reverse()
                return path
            q.append(v)
    return []  # no path
