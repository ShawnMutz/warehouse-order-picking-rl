import numpy as np
from collections import deque


def bfs_shortest_distances(grid, start):
    """
    Compute shortest path distances from start to all traversable cells using BFS.
    """
    H, W = grid.shape
    dist = np.full((H, W), np.inf)
    if grid[start[0], start[1]] != 0:
        return dist
    dist[start] = 0
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0 and np.isinf(dist[nr, nc]):
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def pairwise_distances(grid, points):
    """
    Compute pairwise shortest distances among a list of points (each a tuple (r,c)).
    Returns a square matrix.
    """
    n = len(points)
    dist = np.zeros((n, n))
    for i, p in enumerate(points):
        d = bfs_shortest_distances(grid, p)
        for j in range(i+1, n):
            dist[i, j] = dist[j, i] = d[points[j]]
    return dist


def held_karp_distance(grid, depot, picks):
    """
    Exact optimal route length using Held-Karp dynamic programming.
    All points (depot + picks) must be visited exactly once, returning to depot.
    """
    points = [depot] + list(picks)
    n = len(points)
    dist = pairwise_distances(grid, points)

    # dp[mask][i] = shortest path starting at depot (point 0), visiting all in mask, ending at i
    dp = np.full((1 << n, n), np.inf)
    dp[1][0] = 0.0

    for mask in range(1 << n):
        for u in range(n):
            if not (mask & (1 << u)) or np.isinf(dp[mask, u]):
                continue
            for v in range(n):
                if mask & (1 << v):
                    continue
                new_mask = mask | (1 << v)
                dp[new_mask, v] = min(
                    dp[new_mask, v], dp[mask, u] + dist[u, v])

    full_mask = (1 << n) - 1
    optimal = min(dp[full_mask, u] + dist[u, 0] for u in range(n))
    return optimal
