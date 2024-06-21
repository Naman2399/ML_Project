from typing import List


class Solution:

    def __init__(self):
        self.visit = set()
        return

    def snakesAndLadders(self, board: List[List[int]]) -> int:

        self.res = float('inf')
        board_arr = []

        rows = len(board)
        check = 0
        for i in range(rows - 1, -1, -1):
            if check == 0:
                # Travere l to r
                cols = len(board[i])
                for j in range(cols):
                    board_arr.append(board[i][j])

                check = (check + 1) % 2

            else:
                # Traverse r to l
                cols = len(board[i])
                for j in range(cols - 1, -1, -1):
                    board_arr.append(board[i][j])

        print(board_arr)
        adj = [[] for i in range(len(board_arr))]

        for i in range(len(board_arr) - 1):
            if board_arr[i] == -1:
                for j in range(i + 1, i + 7):
                    if j <= len(board_arr):
                        adj[i].append(j)
                    else:
                        break
            else:
                adj[i].append(board_arr[i])

        for i,row in enumerate(adj) :
            print(row)


        

        return self.res

board = [[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,35,-1,-1,13,-1],[-1,-1,-1,-1,-1,-1],[-1,15,-1,-1,-1,-1]]
solution = Solution()
print(solution.snakesAndLadders(board))

