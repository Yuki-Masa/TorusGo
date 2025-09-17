# ai/MCTS.py

import random
import copy
import time
import math
import numpy as np
from scipy.sparse import csr_matrix
import gc
import os
from matrix_functions.py import*
from PolicyNet.py import*
from ValueNet.py import*

#コミと碁盤サイズ
komi = 5.5
bs = 6

# デバイスの選択
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ネットワークの入力と出力の形状を定義
input_shape = (1, 16, bs, bs)
output_spatial_shape = (bs, bs)

resnet_layers_config = [3, 4, 6, 3]
num_classes_for_resnet = 1000

# Custom3DNetSpatialOutputのインスタンス化
PolicyNet = Custom2DNetSpatialOutput(
    input_shape_original_3d=input_shape,
    output_spatial_shape_2d=output_spatial_shape,
    resnet_layers=resnet_layers_config,
    num_classes_for_resnet=num_classes_for_resnet
).to(device)

if os.path.isfile('/content/drive/MyDrive/TorusGo/model_weight_policy.pth'):
    PolicyNet.load_state_dict(torch.load('/content/drive/MyDrive/TorusGo/model_weight_policy.pth', map_location=device))
else:
    print("No policyweightfile")

# Custom3DNetScalarOutputのインスタンス化
ValueNet = Custom2DNetScalarOutput(
    input_shape_original_3d=input_shape,
    output_spatial_shape_2d=output_spatial_shape,
    resnet_layers=resnet_layers_config,
    num_classes_for_resnet=num_classes_for_resnet
).to(device)

if os.path.isfile('/content/drive/MyDrive/TorusGo/model_weight_value.pth'):
    ValueNet.load_state_dict(torch.load('/content/drive/MyDrive/TorusGo/model_weight_value.pth', map_location=device))
else:
    print("No valueweightfile")


class MCTSNode:
    b_size = 6
    def __init__(self, parent, move, board_state, current_player_color, ko_point):
        self.size = len(board_state)
        self.parent = parent
        self.move = move
        self.children = []
        self.uct = None
        self.policymap = None
        self.visits = 1
        self.unexplored_moves = None
        self.likely_options = None
        self.board_state = copy.deepcopy(board_state) # このノードでの盤面状態
        self.current_player_color = current_player_color # このノードの手番プレイヤー
        self.ko_point = ko_point # このノードでのコウ点 (前回の着手によるコウ点)

    def shape_board2tensor(self,prev_cur_boards,preserve=False):
        """Draft"""
        #相手の色
        enemy_color = 1 if self.current_player_color==2 else 2

        #盤面をレイヤーに分ける
        alies = [[[0 if (value!=self.current_player_color) or (value==0) else 1 for value in row] for row in rowcol] for rowcol in prev_cur_boards]
        enemies = [[[0 if (value==self.current_player_color) or (value==0) else 1 for value in row] for row in rowcol] for rowcol in prev_cur_boards]
        legal_map = [[0 if [i,j] in self.get_legal_moves(self.current_player_color,self.ko_point) else 1 for i in range(b_size)] for j in range(b_size)]
        thumbing_map = [[0 if self.is_self_thumbing(self.current_player_color,i,j) else 1 for i in range(b_size)] for j in range(b_size)]

        #そこに着手した場合の呼吸点のマップ、自分/相手の石の呼吸点のマップ、自分/相手のアタリのマップを初期化
        num_empty_if_put_map = num_empty_map_alies = num_empty_map_enemies = atari_map_alies = atari_map_enemies = [[0 for i in range(b_size)] for j in range(b_size)]

        #そこに着手した場合の呼吸点のマップを作成
        empty_points = get_coords_of_non_value(self.board_state, [self.current_player_color,enemy_color])
        for point in empty_points:
            board = np.array(copy.deepcopy(self.board_state))
            neighbors = np.array(self.get_neighbors(point[0],point[1]))
            indices = tuple(np.transpose(neighbors))
            values_at_coords = board[indices]
            # 抽出された全ての値がtarget_valueと等しいかを確認
            if np.all(values_at_coords == 0):
                num_empty_if_put_map[point[0]][point[1]] = 4
            elif np.all(values_at_coords != self.current_player_color):
                self.place_stone_and_capture(point[0],point[1],self.current_player_color,board=board)
                neighbors = np.array(self.get_neighbors(point[0],point[1]))
                indices = tuple(np.transpose(neighbors))
                values_at_coords = board[indices]
                num_empty_if_put_map[point[0]][point[1]] = np.sum(values_at_coords == 0)
            else:
                board = copy.deepcopy(self.board_state)
                self.place_stone_and_capture(point[0],point[1],self.current_player_color,board=board)
                empty_count = self.get_group_and_liberties(point[0],point[1],self.current_player_color,board=board)
                num_empty_if_put_map[point[0]][point[1]] = len(empty_count[1])

        #自分/相手の石の呼吸点のマップ、自分/相手のアタリのマップを作成
        stone_coords_alies = get_coords_of_non_value(self.board_state, [0,enemy_color])
        stone_coords_enemies = get_coords_of_non_value(self.board_state, [0,self.current_player_color])
        if stone_coords_alies:
            for coord_alies in stone_coords_alies:
                liberties_alies = self.get_group_and_liberties(coord_alies[0],coord_alies[1],self.current_player_color)

                if len(liberties_alies[1])==1:
                    atari_map_alies[list(list(liberties_alies[1])[0])[0]][list(list(liberties_alies[1])[0])[1]] = 1
                for stone in liberties_alies[0]:

                    stone_coords_alies.remove(list(stone))
                    num_empty_map_alies[stone[0]][stone[0]] = len(liberties_alies[1])

        if stone_coords_enemies:
            for coord_enemies in stone_coords_enemies:
                liberties_enemies = self.get_group_and_liberties(coord_enemies[0],coord_enemies[1],enemy_color)

                if len(liberties_enemies[1])==1:
                    #print("libertiesenemies[1]",liberties_enemies[1])
                    atari_map_enemies[list(list(liberties_enemies[1])[0])[0]][list(list(liberties_enemies[1])[0])[1]] = 1
                for stone in liberties_enemies[0]:

                    stone_coords_enemies.remove(list(stone))
                    num_empty_map_enemies[stone[0]][stone[0]] = len(liberties_enemies[1])

        tensor2input = torch.tensor([alies+enemies+[legal_map]+[thumbing_map]+[num_empty_if_put_map]+[num_empty_map_alies]+[num_empty_map_enemies]+[atari_map_alies]+[atari_map_enemies]+[full_array(legal_map,komi/3 if self.current_player_color==2 else -komi/3)]])
        if preserve==False:
            return tensor2input.float().to(device)
        else:
            return tensor2input


    def _get_all_leaf_depths(self, current_depth=0):
        if not self.children:
            # 葉ノードに到達したら、その深さをyield
            yield current_depth
        else:
            # 子がいる場合、各子に対して再帰的に呼び出し、深さを1増やす
            for child in self.children:
                yield from child._get_all_leaf_depths(current_depth + 1)

    def get_tree_metrics(self):
        """
        あるインスタンスから始まるツリーの以下を計算
        1. max_depth (最大の世代深さ)
        2. max_depth_lineages (最大の深さを持つ系統の数)
        3. average_depth (平均世代深さ)
        """
        # 全ての葉ノードの深さをリストとして取得
        all_leaf_depths = list(self._get_all_leaf_depths(current_depth=0))

        if not all_leaf_depths:
            return 0, 1, 0.0 #"max_depth", "max_depth_lineages", "average_depth"


        #最大の世代深さ
        max_depth = max(all_leaf_depths)

        #最大の深さを持つ系統の数
        max_depth_lineages = all_leaf_depths.count(max_depth)

        #平均世代深さ
        average_depth = sum(all_leaf_depths) / len(all_leaf_depths)

        return max_depth, max_depth_lineages, average_depth

    def get_prev_board_states(self,num_generations=3):
        input_data = []
        anscestor = self.parent if self.parent else [[0 for i in range(b_size)] for j in range(b_size)]
        for i in range(num_generations):
            if isinstance(anscestor,list):
                input_data.append(anscestor)
            else:
                input_data.append(anscestor.board_state)
                anscestor = anscestor.parent if anscestor.parent else [[0 for i in range(b_size)] for j in range(b_size)]
        return input_data

    """Draft
    def decompress_board_state(self,root):
        dummy_node = MCTSNode(None,None,root.board_state,root.)
        board = [[0 for i in range(9)] for j in range(9)]
        while move in self.moves:
            move = self.parent.move
            MCTSNode.place_stone_and_capture()
    """

    def uct_value(self, Qvalue,C_param=1.4):
        """UCT (Upper Confidence Bound 1 applied to trees) """
        #if self.visits == 0:
            #return float('inf') # 未探索のノードを優先
        #if self.board_state == self.parent.board_state:
            #print("inf")
            #return float("-inf")
        #input_data = self.parent.shape_board2tensor(self.parent.get_prev_board_states()+[self.parent.board_state])
        #Qvalue = PolicyNet.forward(input_data)[0][self.move[0]][self.move[1]].tolist()

        #親ノードにおいてその手を選択する確率 + 探索パラメーター * sqrt(log(親の訪問回数) / 自分の訪問回数)
        uct = round(Qvalue+C_param * Qvalue *math.sqrt(math.log(self.parent.visits) / (1+self.visits)),2)

        """
        del Qvalue,input_data
        gc.collect()
        if device == torch.device("cuda"):
            torch.cuda.empty_cache()
        """
        #親ノードにおいてその手を選択する確率 + 探索パラメーター * sqrt(log(親の訪問回数) / 自分の訪問回数)
        return  uct

    def is_fully_expanded(self):
        return self.unexplored_moves is not None and len(self.unexplored_moves) == 0

    def add_child(self, move, board_state, ko_point):
        child_node = MCTSNode(self, move, board_state, 1 if self.current_player_color==2 else 2, ko_point)
        self.children.append(child_node)
        return child_node

    def update_stats(self, reward):
        self.visits += 1
        self.uct = reward

    def _wrap_coord(self, coord):
        return (coord % self.size + self.size) % self.size

    def get_efficient_moves(self):
        efficient_moves = []
        input_data = self.shape_board2tensor(self.get_prev_board_states()+[self.board_state])

        #着手確率分布
        q_dist = PolicyNet.forward(input_data)[0].tolist()
        #閾値以上に絞る
        self.likely_options = get_coords_over_threshold(q_dist,0.4)

        #合法手
        legal_moves = self.get_legal_moves(self.current_player_color, self.ko_point)

        #閾値以上且つ合法手 最大8候補
        for max_value in max_in_tensor_generator(q_dist):
            indices=get_indices(q_dist,max_value)
            for index in indices:
                if index in legal_moves:
                    efficient_moves.append(index)
                    if len(efficient_moves)>=min(8,len(legal_moves)):
                        return efficient_moves

        return efficient_moves



    def get_neighbors(self, r, c, diagnal=False):
        """指定された座標の隣接点を取得（トーラス構造を考慮）"""
        neighbors = []
        directions = [
            (-1, 0), (1, 0), # 上下
            (0, -1), (0, 1)  # 左右
        ]

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            nr = self._wrap_coord(nr)
            nc = self._wrap_coord(nc)
            neighbors.append((nr, nc))

        if diagnal == False:
            return neighbors
        else:
            neighbors_diag = []
            directions_diag = [
                (-1, 1), (-1,-1),
                (1, 1), (1, -1)
            ]
            for dr, dc in directions_diag:
                nr, nc = r + dr, c + dc
                nr = self._wrap_coord(nr)
                nc = self._wrap_coord(nc)
                neighbors_diag.append((nr, nc))


            return neighbors,neighbors_diag

    def get_group_and_liberties(self, r, c, color, board=None, visited=None):
        """グループと呼吸点を取得"""
        if visited is None:
            visited = set()
        if board is None:
            board = self.board_state

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
                    neighbor_color = board[nr][nc]

                    if neighbor_color == color:
                        stack.append((nr, nc))
                    elif neighbor_color == 0:
                        liberties.add((nr, nc))
        return group, liberties

    def get_territories(self):
        """to get coords of territories
        territory: [[x,y],[],...] coords of a connected territory
        territories:[group1,group2,...] set of connected teritories
        """
        empty_points = [[i,j] for i,j in range_ndim(b_size,b_size) if self.board_state[i][j]==0]
        black_territories = []
        white_territories = []

        for point in empty_points:
            territory = set()
            stack = [point]
            surrounding_color = 3 #any value /0,1,2 is ok
            while stack:
                curr_r, curr_c = stack.pop()
                territory.add((curr_r, curr_c))
                #legal_points.remove((curr_r, curr_c))

                for nr, nc in self.get_neighbors(curr_r, curr_c):
                    #if ((nr, nc) in legal_points) and ((nr, nc) not in atari_points):
                    if [nr, nc] in empty_points:
                        stack.append([nr, nc])
                        empty_points.remove([nr, nc])
                    elif surrounding_color == 3:
                        surrounding_color = self.board_state[nr][nc]
                    elif is_territory and self.board_state[nr][nc] != surrounding_color:
                        is_territory = False
            if is_territory:
                if surrounding_color == 1:
                    black_territories.append(territory)
                else:
                    white_territories.append(territory)
        return [black_territories, white_territories]

    def place_stone_and_capture(self, r, c, color,board=None):
        """石を置き、取れる石があれば取り、取られた石を返す。自殺手ならNone"""
        if board is None:
            board = self.board_state

        if board[r][c] != 0:
            return None # 既に石がある

        board[r][c] = color
        opponent_color = 1 if color == 2 else 2
        removed_stones = []

        # 相手の石が取られるかチェック
        for nr, nc in self.get_neighbors(r, c):
            if board[nr][nc] == opponent_color:
                group, liberties = self.get_group_and_liberties(nr, nc, opponent_color,board=board)
                if len(liberties) == 0:
                    for rr, cc in list(group): # Setをlistに変換してイテレート
                        removed_stones.append((rr, cc))
                        board[rr][cc] = 0

        # 自殺手チェック (相手の石を取っていない場合のみ)
        if not removed_stones:
            group, liberties = self.get_group_and_liberties(r, c, color,board=board)
            if len(liberties) == 0:
                board[r][c] = 0 # 自殺手なので元に戻す
                return None # 自殺手
        """
        else:
            self.prev_board_states.append(self.board_state)
            del self.prev_board_states[0]
        """

        return removed_stones

    def is_valid_move(self, r, c, color, ko_point):
        """有効な手か判定 (コウ、自殺手、埋まっている場所)"""
        if not (0 <= r < self.size and 0 <= c < self.size): return False
        if self.board_state[r][c] != 0: return False

        temp_board = [row[:] for row in self.board_state] # 盤面をディープコピー

        if [r,c] == ko_point:
            return False

        removed_stones = self.place_stone_and_capture(r, c, color)
        if removed_stones is None: # 自殺手
            self.board_state = [row[:] for row in temp_board] # 元に戻す
            return False

        self.board_state = [row[:] for row in temp_board] # 元に戻す
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

    def get_legal_moves(self, player_color,ko_point,selfplay=True):
        """合法手リストを取得（パスを含む）　目潰し手も除いている"""
        legal_moves = []
        for r in range(self.size):
            for c in range(self.size):
                if self.is_valid_move(r, c, player_color, ko_point):
                    if not selfplay or not self.is_self_thumbing(player_color,r,c): #when selfplay it's needed for preventing endless game
                        legal_moves.append([r, c])

        legal_moves.append("pass") # パスも合法手として含める
        return legal_moves

    def count_stones(self, komi=6.5):
        """終局時の盤上の石数を数える（簡易的な地合い考慮なし）"""
        black_stones = 0
        white_stones = 0
        for r in range(self.size):
            for c in range(self.size):
                if self.board_state[r][c] == 1:
                    black_stones += 1
                elif self.board_state[r][c] == 2:
                    white_stones += 1
        return black_stones, white_stones + komi

    def _is_game_over(self):
        legal_moves1 = self.get_legal_moves(self.current_player_color,self.ko_point)
        legal_moves2 = self.get_legal_moves(1 if self.current_player_color==2 else 2,None)
        if not legal_moves1 and not legal_moves2:
            return True
        """
        elif self.board_state.count(0) < 81/6:
            black_territories,white_territories = self.get_territories()
            if (not [b for b in black_territories if len(b)>6]) and (not [w for w in white_territories if len(w)>6]):
                return True
        """
        return False

class Engine:
    b_size = 6
    def __init__(self, parent=None, move=None, board_state=[[0 for i in range(b_size)] for j in range(b_size)],\
                current_player_color=1, ko_point=None, num_simulations=1500): # 試行回数を調整
        self.num_simulations = num_simulations
        self.root =  MCTSNode(
            parent, move,
            board_state,
            current_player_color,
            ko_point # ルートノードのコウ点は現在のコウ点
        )

    def find_best_move(self): #複数通信の挙動を見るときは引数に+game_id
        """MCTSを実行し、最適な手を返す"""

        """外部から現在の盤面状態をロード"""
        """
        if board_state != self.root.board_state or prev_boards != self.root.prev_board_states \
        or current_player_color!=self.root.current_player_color or ko_point!=self.root.ko_point:
            self.root.__init__(
                None, None,
                board_state,
                current_player_color,
                tuple(ko_point) if ko_point else None, # ルートノードのコウ点は現在のコウ点
                prev_boards # ルートノードのprev_board_stateは現在のボード状態
            )
        """

        #start_time = time.time() # 探索開始時刻を記録

        for i in range(self.num_simulations):
            node = self._select(self.root) # 選択と展開
            #get_max_depth_lambda = lambda instance: 1 + max(get_max_depth_lambda(child) for child in instance.children) if instance.children else 0
            #print("num",i,"depth",get_max_depth_lambda(self.root))

            if i%50==0:
                tree_metrics = self.root.get_tree_metrics()
                print("num",i,"max depth",tree_metrics[0],"lineages over max depth",tree_metrics[1],"average_depth",tree_metrics[2])
            if node is None: # 選択できるノードがない場合（例：ゲーム終了）
                print("end")
                continue
            else:
                if i%50==0:
                    print("move:",node.move,"turn:",node.current_player_color)
                    #calculate_descendants_lambda = lambda inst: sum(1 + calculate_descendants_lambda(child) for child in inst.children) if inst.children else 0
                    #print("depth",calculate_descendants_lambda(self.root))

            input_data = node.shape_board2tensor(node.get_prev_board_states()+[node.board_state])
            evaluation = ValueNet.forward(input_data)
            self._backpropagate(node, evaluation) # 逆伝播
            del evaluation,input_data
            gc.collect()
            if device == torch.device("cuda"):
                torch.cuda.empty_cache()

            """
            # 進捗状況と残り時間の概算を出力
            if (i + 1) % 100 == 0 or (i + 1) == self.num_simulations: # 50回ごとに、または最後にログ出力
                elapsed_time = time.time() - start_time
                simulations_per_second = (i + 1) / elapsed_time
                remaining_simulations = self.num_simulations - (i + 1)
                estimated_remaining_time = remaining_simulations / simulations_per_second

                #print(f"GameID: {game_id}")
                print(f"シミュレーション進捗: {i + 1}/{self.num_simulations}完了")
                print(f"経過時間: {elapsed_time:.2f}秒")
                print(f"残りシミュレーション数: {remaining_simulations}")
                print(f"推定残り時間: {estimated_remaining_time:.2f}秒")
                print("-" * 30)
            """

        # 最も訪問回数の多い子ノードを選択
        if not self.root.children:
            return None, None # 合法手がない場合（パスを返すなどの処理も考慮）
        else:
            best_child = max(self.root.children, key=lambda child: child.visits)

        p_dist = list(map(lambda x:x/np.sum(dist), dist:=[[0.01 if [i,j] not in [r.move for r in self.root.children] else \
        next(child.visits for child in self.root.children if child.move == [i,j]) for i in range(b_size)] for j in range(b_size)]))

        if best_child.move == "pass":
            return None, None

        return best_child.move[0], best_child.move[1], p_dist

    def _select(self, node):
        """選択フェーズ: 未探索ノードが見つかるか、ゲームが終了するまでUCTを使ってノードを選択"""
        #while node.is_fully_expanded() and node.children and node.visits>1:
        #to go deeper until no childnode or (be unexploredmove and no likelyoptions) or never visited before
        while node.children and (node.is_fully_expanded() or node.likely_options) \
         and node.visits>2:
            #node = max(node.children, key=lambda child: child.uct_value())
            node = sorted(node.children, key=lambda child: child.uct)[-1]

            # 簡易的なゲーム終了判定 (例: 連続パスなど、シミュレーションで判断させるため、ここでは厳密には不要)
            if node._is_game_over():
                print(node.board_state)
                scores=node.count_stones()
                self._backpropagate(node,1 if scores[node.current_player_color-3]>scores[node.current_player_color-2] else -1)
                node = sorted(node.parent.children, key=lambda child: child.uct)[-2]

        # 展開フェーズ: 選択されたノードから未探索の子供を1つ追加
        if node.policymap is None:
            input_data = node.shape_board2tensor(node.get_prev_board_states()+[node.board_state])
            node.policymap = PolicyNet.forward(input_data)[0].tolist()
            del input_data
            gc.collect()
            if device == torch.device("cuda"):
                torch.cuda.empty_cache()

        """
        if node.unexplored_moves is None:
            node.unexplored_moves = node.get_legal_moves(node.current_player_color, node.ko_point)
            random.shuffle(node.unexplored_moves)
        """
        if node.unexplored_moves is None:
            node.unexplored_moves = node.get_efficient_moves()
            if len(node.unexplored_moves)==0:
                node.unexplored_moves = node.get_legal_moves(node.current_player_color, node.ko_point)
            random.shuffle(node.unexplored_moves)
        #print("middle",node.unexplored_moves)
        if node.unexplored_moves:
            move = node.unexplored_moves.pop() # 未探索の手を1つ取り出す
            next_node = node.add_child(move, node.board_state, node.ko_point)

            next_node.uct = next_node.uct_value(next_node.parent.policymap[next_node.move[0]][next_node.move[1]])
            new_ko_point = None
            if move != "pass":
                removed_stones = next_node.place_stone_and_capture(move[0], move[1], node.current_player_color)
                # コウの更新 (単一の石を取り、かつその手で取った石が単一の石で、かつ自分の石の呼吸点が1つになった場合)
                # 複雑なコウの正確な判定は、サーバー側GoBoard.jsと完全に一致させる必要があります
                if removed_stones and len(removed_stones) == 1 and \
                   next_node.get_group_and_liberties(move[0], move[1], node.current_player_color)[0].issubset({(move[0], move[1])}) and \
                   len(next_node.get_group_and_liberties(move[0], move[1], node.current_player_color)[1]) == 1:
                    new_ko_point = removed_stones[0]

            next_node_color = 1 if node.current_player_color == 2 else 2
            next_node.current_player_color = next_node_color
            if next_node._is_game_over():
                print(next_node.board_state)
                scores=next_node.count_stones()
                self._backpropagate(node,1 if scores[self.root.current_player_color-3]>scores[self.root.current_player_color-2] else -1)
                return None

            return next_node
        return None

    def _backpropagate(self, node, eval):
        """逆伝播フェーズ: シミュレーション結果をルートまで伝播させる"""
        while node is not None:
            node.update_stats(eval)
            node = node.parent
            # 親ノードの報酬は子のノードの報酬の逆
            eval = -eval
