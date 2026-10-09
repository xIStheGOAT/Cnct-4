from functools import lru_cache
import time

ROWS, COLS = 6, 7
ORDER = (3, 2, 4, 1, 5, 0, 6)

def valid_moves(board):
    return [c for c in ORDER if board[0][c] == 0]

def winner(board):
    for r in range(ROWS):
        for c in range(COLS):
            p = board[r][c]
            if not p:
                continue
            if c <= COLS-4 and all(board[r][c+i] == p for i in range(4)):
                return p
            if r <= ROWS-4 and all(board[r+i][c] == p for i in range(4)):
                return p
            if r <= ROWS-4 and c <= COLS-4 and all(board[r+i][c+i] == p for i in range(4)):
                return p
            if r <= ROWS-4 and c >= 3 and all(board[r+i][c-i] == p for i in range(4)):
                return p
    return 0

def drop(board, col, player):
    b = [row[:] for row in board]
    for r in range(ROWS-1, -1, -1):
        if b[r][col] == 0:
            b[r][col] = player
            return b
    return None

def is_full(board):
    return all(board[0][c] != 0 for c in range(COLS))

def score_window(vals, me, opp):
    m = vals.count(me)
    o = vals.count(opp)
    e = vals.count(0)
    if m and o:
        return 0
    if m == 4:
        return 100000
    if m == 3 and e == 1:
        return 100
    if m == 2 and e == 2:
        return 10
    if o == 3 and e == 1:
        return -120
    if o == 2 and e == 2:
        return -12
    return 0

def heuristic(board, me):
    opp = 3 - me
    s = 0
    # Center control
    s += sum(6 if board[r][3] == me else -6 if board[r][3] == opp else 0 for r in range(ROWS))
    for r in range(ROWS):
        for c in range(COLS-3):
            s += score_window([board[r][c+i] for i in range(4)], me, opp)
    for c in range(COLS):
        for r in range(ROWS-3):
            s += score_window([board[r+i][c] for i in range(4)], me, opp)
    for r in range(ROWS-3):
        for c in range(COLS-3):
            s += score_window([board[r+i][c+i] for i in range(4)], me, opp)
    for r in range(ROWS-3):
        for c in range(3, COLS):
            s += score_window([board[r+i][c-i] for i in range(4)], me, opp)
    return s

def board_key(board):
    return tuple(x for row in board for x in row)

def best_move(board, player=1, time_limit=4.0):
    start = time.time()
    moves = valid_moves(board)
    if not moves:
        return None, "draw", 0

    # Immediate win.
    for c in moves:
        b = drop(board, c, player)
        if winner(b) == player:
            return c, "winning move", 1

    # Immediate block.
    opp = 3 - player
    for c in moves:
        b = drop(board, c, opp)
        if winner(b) == opp:
            return c, "must block", 1

    cache = {}

    def negamax(b, turn, depth, alpha, beta, root_player):
        if time.time() - start >= time_limit:
            raise TimeoutError
        w = winner(b)
        if w == root_player:
            return 1000000 + depth
        if w == 3-root_player:
            return -1000000 - depth
        if is_full(b):
            return 0
        if depth == 0:
            return heuristic(b, root_player)

        key = (board_key(b), turn, depth, root_player)
        if key in cache:
            return cache[key]

        best = -10**18
        for c in valid_moves(b):
            child = drop(b, c, turn)
            val = negamax(child, 3-turn, depth-1, -beta, -alpha, root_player)
            # Convert to root perspective by evaluating directly rather than sign flip.
            # negamax above is root-perspective; maximize on root's turn, minimize otherwise.
            if turn == root_player:
                best = max(best, val)
                alpha = max(alpha, val)
            else:
                best = min(best, val)
                beta = min(beta, val)
            if alpha >= beta:
                break
        cache[key] = best
        return best

    chosen = moves[0]
    chosen_score = -10**18
    completed = 0

    # Iterative deepening. The returned move is always legal.
    for depth in range(1, 15):
        try:
            local_best = chosen
            local_score = -10**18 if player == 1 else 10**18
            for c in moves:
                b = drop(board, c, player)
                val = negamax(b, 3-player, depth-1, -10**18, 10**18, player)
                if val > local_score:
                    local_score, local_best = val, c
            chosen, chosen_score = local_best, local_score
            completed = depth
        except TimeoutError:
            break

    if completed == 0:
        return chosen, "best quick move", 0

    return chosen, f"best move found at search depth {completed}", completed
