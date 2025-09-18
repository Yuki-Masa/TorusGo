# ai/go_engine_simple.py

import random
import copy
import time
import math

class GoBoardAI:
    def __init__(self, size, torus_type='none'):
        self.size = size
        self.board = [[0 for _ in range(size)] for _ in range(size)] # 0:空, 1:黒, 2:白
        self.torus_type = torus_type
        self.ko_point = None

    def set_board_state(self, board_state, ko_point):
        """外部から盤面状態をロードする"""
        self.board = [row[:] for row in board_state] # ディープコピー
        self.ko_point = tuple(ko_point) if ko_point else None

    def _wrap_coord(self, coord):
        return (coord % self.size + self.size) % self.size

    def get_neighbors(self, r, c):
        """指定された座標の隣接点を取得（トーラス構造を考慮）"""
        neighbors = []
        directions = [
            (-1, 0), (1, 0), # 上下
            (0, -1), (0, 1)  # 左右
        ]

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if self.torus_type == 'all':
                nr = self._wrap_coord(nr)
                nc = self._wrap_coord(nc)
            elif self.torus_type == 'horizontal':
                nc = self._wrap_coord(nc)
                if not (0 <= nr < self.size): continue # 行はラップしない
            else: # 'none' (通常の碁盤)
                if not (0 <= nr < self.size and 0 <= nc < self.size): continue

            neighbors.append((nr, nc))
        return neighbors

    def get_group_and_liberties(self, r, c, color, visited=None):
        """グループと呼吸点を取得"""
        if visited is None:
            visited = set()

        group = set()
        liberties = set()
        stack = [(r, c)]
        visited.add((r, c))

        while stack:
            curr_r, curr_c = stack.pop()
            group.add((curr_r, curr_c))

            for nr, nc in self.get_neighbors(curr_r, curr_c):
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    neighbor_color = self.board[nr][nc]

                    if neighbor_color == color:
                        stack.append((nr, nc))
                    elif neighbor_color == 0:
                        liberties.add((nr, nc))
        return group, liberties

    def get_territories(self, legal_moves):
        """to get coords of territories
        territory: [[x,y],[],...] coords of a connected territory
        territories:[group1,group2,...] set of connected teritories
        """
        legal_points = legal_moves
        black_territories = []
        white_territories = []

        for point in legal_points:
            territory = set()
            stack = [point]
            surrounding_color = 3 #any value /0,1,2 is ok
            while stack:
                curr_r, curr_c = stack.pop()
                territory.add((curr_r, curr_c))
                #legal_points.remove((curr_r, curr_c))

                for nr, nc in self.get_neighbors(curr_r, curr_c):
                    #if ((nr, nc) in legal_points) and ((nr, nc) not in atari_points):
                    if (nr, nc) in legal_points:
                        stack.append((nr, nc))
                        legal_points.remove((nr, nc))
                        #neighbor_color = self.board[nr][nc]
                    elif surrounding_color == 3:
                        #surrounding_color = neighbor_color
                        surrounding_color = self.board[nr][nc]
                    #elif neighbor_color != surrounding_color:
                    elif self.board[nr][nc] != surrounding_color:
                        stack = None
                        break
            if surrounding_color == 1:
                black_territories.append(territory)
            else:
                white_territories.append(territory)
        return [black_territories, white_territories]

    def place_stone_and_capture(self, r, c, color):
        """石を置き、取れる石があれば取り、取られた石を返す。自殺手ならNone。"""
        if self.board[r][c] != 0:
            return None # 既に石がある

        self.board[r][c] = color
        opponent_color = 1 if color == 2 else 2
        removed_stones = []

        # 相手の石が取られるかチェック
        for nr, nc in self.get_neighbors(r, c):
            if self.board[nr][nc] == opponent_color:
                group, liberties = self.get_group_and_liberties(nr, nc, opponent_color)
                if len(liberties) == 0:
                    for rr, cc in list(group): # Setをlistに変換してイテレート
                        removed_stones.append((rr, cc))
                        self.board[rr][cc] = 0

        # 自殺手チェック (相手の石を取っていない場合のみ)
        if not removed_stones:
            group, liberties = self.get_group_and_liberties(r, c, color)
            if len(liberties) == 0:
                self.board[r][c] = 0 # 自殺手なので元に戻す
                return None # 自殺手

        return removed_stones

    def is_valid_move(self, r, c, color, current_ko_point, prev_board_state):
        """有効な手か判定 (コウ、自殺手、埋まっている場所)"""
        if not (0 <= r < self.size and 0 <= c < self.size): return False
        if self.board[r][c] != 0: return False

        # コウの直後禁止
        if current_ko_point and current_ko_point == (r, c):
            return False

        temp_board = [row[:] for row in self.board] # 盤面をディープコピー

        removed_stones = self.place_stone_and_capture(r, c, color)
        if removed_stones is None: # 自殺手
            self.board = [row[:] for row in temp_board] # 元に戻す
            return False

        # コウ判定 (現在の盤面が直前の盤面と全く同じ場合)
        # ただし、取られた石がない場合はコウではない (囲碁のコウの定義)
        if removed_stones and prev_board_state and self.board == prev_board_state:
            self.board = [row[:] for row in temp_board] # 元に戻す
            return False

        self.board = [row[:] for row in temp_board] # 元に戻す
        return True

    def is_self_thumbing(self,player_color,r,c):
        count=0
        count_diag=[0,0,0]
        for neighbor, neighbor_diag in zip(*self.get_neighbors(r,c,True)):
            if self.board_state[neighbor[0]][neighbor[1]]==player_color:
                count+=1
            diag_color=self.board_state[neighbor_diag[0]][neighbor_diag[1]]
            count_diag[diag_color]+=1
        if count == 4 and (count_diag[player_color] >= 3 or (count_diag[player_color] >= 1 and count_diag[0]>=2)):
            return True
        else:
            return False

    def get_available_moves(self, player_color,ko_point,selfplay=True):
        """合法手リストを取得（パスを含む）　目潰し手も除いている"""
        legal_moves = []
        for r in range(self.size):
            for c in range(self.size):
                if self.is_valid_move(r, c, player_color, ko_point):
                    if not selfplay or not self.is_self_thumbing(player_color,r,c): #when selfplay it's needed for preventing endless game
                        legal_moves.append([r, c])

        legal_moves.append("pass") # パスも合法手として含める
        return legal_moves

    def get_legal_moves(self, current_player_color, current_ko_point, prev_board_state):
        """合法手リストを取得（パスを含む）"""
        legal_moves = []
        for r in range(self.size):
            for c in range(self.size):
                if self.is_valid_move(r, c, current_player_color, current_ko_point, prev_board_state):
                    legal_moves.append((r, c))
        #legal_moves.append("pass") # パスも合法手として含める
        return legal_moves


    def count_stones(self, komi=6.5):
        """終局時の盤上の石数を数える（簡易的な地合い考慮なし）"""
        black_stones = 0
        white_stones = 0
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == 1:
                    black_stones += 1
                elif self.board[r][c] == 2:
                    white_stones += 1
        return black_stones, white_stones + komi


class MCTSNode:
    def __init__(self, parent, move, board_state, current_player_color, ko_point, prev_board_state):
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.unexplored_moves = None
        self.board_state = board_state # このノードでの盤面状態
        self.current_player_color = current_player_color # このノードの手番プレイヤー
        self.ko_point = ko_point # このノードでのコウ点 (前回の着手によるコウ点)
        self.prev_board_state = prev_board_state # コウ判定のための直前の盤面状態

    def uct_value(self, C_param=1.4):
        """UCT (Upper Confidence Bound 1 applied to trees) 値を計算"""
        if self.visits == 0:
            return float('inf') # 未探索のノードを優先

        # 勝利数 / 訪問回数 + 探索パラメーター * sqrt(log(親の訪問回数) / 自分の訪問回数)
        return (self.wins / self.visits) + C_param * math.sqrt(math.log(self.parent.visits) / self.visits)

    def is_fully_expanded(self):
        return self.unexplored_moves is not None and len(self.unexplored_moves) == 0

    def add_child(self, move, board_state, next_player_color, ko_point, prev_board_state):
        child_node = MCTSNode(self, move, board_state, next_player_color, ko_point, prev_board_state)
        self.children.append(child_node)
        return child_node

    def update_stats(self, reward):
        self.visits += 1
        self.wins += reward


class GoEngine:
    def __init__(self, board_size, torus_type='all', num_simulations=5000): # 試行回数を調整
        self.board_size = board_size
        self.torus_type = torus_type
        self.num_simulations = num_simulations
        self.current_board = None
        self.current_ko_point = None
        self.previous_board_state = None # コウ判定用

    def set_board_state(self, board_state, ko_point):
        """外部から現在の盤面状態をロード"""
        self.current_board = GoBoardAI(self.board_size, self.torus_type)
        # コウ判定のために、現在のボード状態をprev_board_stateとして保存
        self.previous_board_state = [row[:] for row in board_state]
        self.current_board.set_board_state(board_state, ko_point)
        self.current_ko_point = tuple(ko_point) if ko_point else None

    def find_best_move(self, current_player_color,game_id):
        """MCTSを実行し、最適な手を返す"""
        root = MCTSNode(
            None, None,
            [row[:] for row in self.current_board.board],
            current_player_color,
            self.current_ko_point, # ルートノードのコウ点は現在のコウ点
            self.previous_board_state # ルートノードのprev_board_stateは現在のボード状態
        )

        start_time = time.time() # 探索開始時刻を記録

        for i in range(self.num_simulations):
            node = self._select(root) # 選択と展開
            if node is None: # 選択できるノードがない場合（例：ゲーム終了）
                continue
            winner_color = self._simulate(node) # シミュレーション
            self._backpropagate(node, winner_color) # 逆伝播

            # 進捗状況と残り時間の概算を出力
            if (i + 1) % 200 == 0 or (i + 1) == self.num_simulations: # 50回ごとに、または最後にログ出力
                elapsed_time = time.time() - start_time
                simulations_per_second = (i + 1) / elapsed_time
                remaining_simulations = self.num_simulations - (i + 1)
                estimated_remaining_time = remaining_simulations / simulations_per_second

                print(f"GameID: {game_id}")
                print(f"シミュレーション進捗: {i + 1}/{self.num_simulations}完了")
                print(f"経過時間: {elapsed_time:.2f}秒")
                print(f"残りシミュレーション数: {remaining_simulations}")
                print(f"推定残り時間: {estimated_remaining_time:.2f}秒")
                print("-" * 30)

        # 最も訪問回数の多い子ノードを選択
        if not root.children:
            return None, None # 合法手がない場合（パスを返すなどの処理も考慮）

        if  sum(self.previous_board_state,[]).count(0)>(self.board_size^2)/6:
            if self.torus_type=='horizontal':
                nchildren = [q for q in root.children if q.move[0]>1 and q.move[0]<7]
            elif self.torus_type=="none":
                nchildren = [q for q in root.children if q.move[0]>1 and q.move[0]<7 and q.move[1]>1 and q.move[1]<7]
            else:
                nchildren = root.children
            best_child = max(nchildren, key=lambda child: child.visits)
        else:
            best_child = max(root.children, key=lambda child: child.visits)


        if best_child.move == "pass":
            return None, None

        return best_child.move[0], best_child.move[1]

    def _select(self, node):
        """選択フェーズ: 未探索ノードが見つかるか、ゲームが終了するまでUCTを使ってノードを選択"""
        #while node.is_fully_expanded() and node.children:
        while node.children and node.is_fully_expanded() and node.visits>2:
            #node = max(node.children, key=lambda child: child.uct_value())
            node = sorted(node.children, key=lambda child: child.uct)[-1]

            # 簡易的なゲーム終了判定 (例: 連続パスなど、シミュレーションで判断させるため、ここでは厳密には不要)
            # if self._is_game_over_state(node.board_state): return node

        # 展開フェーズ: 選択されたノードから未探索の子供を1つ追加
        if node.unexplored_moves is None:
            temp_board_ai = GoBoardAI(self.board_size, self.torus_type)
            temp_board_ai.set_board_state(node.board_state, node.ko_point)
            node.unexplored_moves = temp_board_ai.get_available_moves(node.current_player_color, node.ko_point, node.prev_board_state)
            random.shuffle(node.unexplored_moves)

        if node.unexplored_moves:
            move = node.unexplored_moves.pop() # 未探索の手を1つ取り出す

            next_board_ai = GoBoardAI(self.board_size, self.torus_type)
            next_board_ai.set_board_state(node.board_state, node.ko_point) # 親のko_pointを初期値としてセット

            new_ko_point = None
            if move != "pass":
                removed_stones = next_board_ai.place_stone_and_capture(move[0], move[1], node.current_player_color)
                # コウの更新 (単一の石を取り、かつその手で取った石が単一の石で、かつ自分の石の呼吸点が1つになった場合)
                if removed_stones and len(removed_stones) == 1 and \
                   next_board_ai.get_group_and_liberties(move[0], move[1], node.current_player_color)[0].issubset({(move[0], move[1])}) and \
                   len(next_board_ai.get_group_and_liberties(move[0], move[1], node.current_player_color)[1]) == 1:
                    new_ko_point = removed_stones[0]

            next_player_color = 1 if node.current_player_color == 2 else 2

            return node.add_child(move, next_board_ai.board, next_player_color, new_ko_point, [row[:] for row in node.board_state]) # prev_board_stateを渡す
        return node

    def _simulate(self, node):
        #シミュレーションフェーズ: ランダムプレイアウトで勝者を予測
        temp_board_ai = GoBoardAI(self.board_size, self.torus_type)
        temp_board_ai.set_board_state(node.board_state, node.ko_point)

        current_player = node.current_player_color
        current_ko_point = node.ko_point
        prev_board_state_sim = [row[:] for row in node.board_state] # シミュレーション中のコウ判定用
        pass_count_simulation = 0
        max_moves_in_sim = self.board_size * self.board_size * 2 # 無限ループ回避

        for _ in range(max_moves_in_sim):
            legal_moves = temp_board_ai.get_available_moves(current_player, current_ko_point, prev_board_state_sim)

            if not legal_moves:
                move = "pass"
            elif prev_board_state_sim.count(0) >= 5*max_moves_in_sim/6:
                if self.torus_type=="horizontal":
                    move = random.choice([q for q in legal_moves if q[0]>1 and q[0]<7])
                #if self.torus_type=="all":
                    #move = random.choice([q for q in legal_moves if q[0]>1 and q[0]<7] and q[1]>1 and q[1]<7)
                else:
                    move = random.choice(legal_moves)
            else:
                move = random.choice(legal_moves)
                if prev_board_state_sim.count(0) < max_moves_in_sim/10:
                    black_territories,white_territories = temp_board_ai.get_territories(legal_moves)
                    if (not [b for b in black_territories if len(b)>6]) and (not [w for w in white_territories if len(w)>6]):
                        move = "pass"
                #else:
                    #move = random.choice(legal_moves)

            if move == "pass":
                pass_count_simulation += 1
                if pass_count_simulation >= 2:
                    break # 2回連続パスでシミュレーション終了
                current_ko_point = None # パス後はコウ点クリア
            else:
                pass_count_simulation = 0
                prev_board_state_sim = [row[:] for row in temp_board_ai.board] # 次のコウ判定のために現在の盤面を保存
                removed_stones = temp_board_ai.place_stone_and_capture(move[0], move[1], current_player)

                # シミュレーション中のコウ点更新
                if removed_stones and len(removed_stones) == 1 and \
                   temp_board_ai.get_group_and_liberties(move[0], move[1], current_player)[0].issubset({(move[0], move[1])}) and \
                   len(temp_board_ai.get_group_and_liberties(move[0], move[1], current_player)[1]) == 1:
                    current_ko_point = removed_stones[0]
                else:
                    current_ko_point = None

            current_player = 1 if current_player == 2 else 2 # 手番を交代

        black_score, white_score = temp_board_ai.count_stones()

        # ルートノードの手番プレイヤー (AIが打つ番) が勝ったかどうかを報酬とする
        # AIが黒番(1)で勝ち => 1
        # AIが白番(2)で勝ち => 1
        # それ以外 (負け/引き分け) => 0
        #if current_player_color == 1 and black_score > white_score:
        if current_player == 1 and black_score > white_score:
            return 1
        #elif current_player_color == 2 and white_score > black_score:
        elif current_player == 2 and white_score > black_score:
            return 1
        elif black_score == white_score: # 引き分け
            return 0.5
        return 0 # 負け

    def _backpropagate(self, node, winner_color):
        """逆伝播フェーズ: シミュレーション結果をルートまで伝播させる"""
        while node is not None:
            node.update_stats(winner_color)
            node = node.parent
            # 親ノードの報酬は子のノードの報酬の逆
            if node and winner_color in [1, 2]: # 引き分けでない場合
                 winner_color = 1 if winner_color == 2 else 2 # 相手の報酬
            elif node and winner_color == 0.5: # 引き分けの場合、報酬は0.5のまま
                pass
