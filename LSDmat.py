import numpy as np

Msg = "1100000001101000111110100000101000111101000110010000010101100001"
M = [None, Msg[:16], Msg[16:32], Msg[32:48], Msg[48:64]]
LSD = [[1, 2, 3, 4], [3, 4, 1, 2], [2, 1, 4, 3], [4, 3, 2, 1]]
LSD_map = np.zeros((16, 16))
for i in range(4):
    for j in range(4):
        msg = M[LSD[i][j]]
        for x in range(4):
            for y in range(4):
                LSD_map[i * 4 + x, j * 4 + y] = msg[x * 4 + y]
print(LSD_map)

