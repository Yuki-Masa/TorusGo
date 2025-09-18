#selfplay2learn.py

import pickle
import torch
import gc
import os
board_size = 6
input_shape = (1,16, board_size, board_size)
output_spatial_shape = (board_size, board_size)

resnet_layers_config = [3, 4, 6, 3]
num_classes_for_resnet = 1000

dataset = []
for i in range(1):
    data = [[],[]]
    player = 1
    kopoint = None
    total_moves = 0
    interval = 0
    prevboard_matrix = [[[0 for i in range(board_size)] for j in range(board_size)] for k in range(3)]

    if os.path.isfile('/content/drive/MyDrive/TorusGo/data.pickle'):
        with open('/content/drive/MyDrive/TorusGo/data.pickle', mode='br') as fi:
            data = pickle.load(fi)

        #ロードした対局途中のデータから現在の手番を求める
        if len(data[0])>len(data[1]):
            player = 1
        else:
            player = 2

        #ロードした対局途中のデータから現在の盤面を求める
        alies_layer = np.array(data[player-1][-1][0][0][0]).astype(int)
        enemies_layer = np.array(data[player-1][-1][0][0][4]).astype(int)
        board = (alies_layer+2*enemies_layer).tolist() if player == 1 else (2*alies_layer+enemies_layer).tolist()

        dummyenemy = 1 if player==2 else 2
        dummyplayer = 1 if player==1 else 2
        for i in range(min(3,len(data[0])+len(data[1])-1)):
            prev_alies_layer = np.array(data[dummyenemy-1][-1-i][0][0][0])
            prev_enemies_layer = np.array(data[dummyenemy-1][-1-i][0][0][1])
            prevboard = (prev_alies_layer+2*prev_enemies_layer).tolist() if dummyenemy == 1 else (2*prev_alies_layer+prev_enemies_layer).tolist()
            prevboard_matrix[i] = prevboard
            dummyenemy = 1 if dummyenemy==2 else 2
            dummyplayer = 1 if dummyplayer==2 else 2

    if os.path.isfile('/content/drive/MyDrive/TorusGo/kolog.pickle'):
        with open('/content/drive/MyDrive/TorusGo/kolog.pickle', mode='br') as fi:
            ko = pickle.load(fi)
            ko_point = ko
        ai = Engine(board_state=board,current_player_color=player,ko_point=ko_point)
        print(board)
    else:
        ai = Engine()

    while not ai.root._is_game_over():
        best_move_row, best_move_col,p_dist = ai.find_best_move()
        total_moves += 1
        interval += 1
        print("total:",total_moves)
        print("bestmove",best_move_row,best_move_col)

        data[player-1].append([ai.root.shape_board2tensor(prevboard_matrix+[ai.root.board_state],preserve=True), p_dist, player])
        prevboard_matrix.insert(0,ai.root.board_state)
        del prevboard_matrix[2]

        if not (best_move_row == None):
            removed = ai.root.place_stone_and_capture(best_move_row, best_move_col,player)
            if removed and len(removed) == 1 and \
               ai.root.get_group_and_liberties(best_move_row, best_move_col, player)[0].issubset({(best_move_row, best_move_col)}) and \
               len(ai.root.get_group_and_liberties(best_move_row, best_move_col, 1)[1]) == 1:
                ko_point = removed
            else:
                ko_point = None

        print(ai.root.board_state)

        player = 1 if player==2 else 2
        next_board = ai.root.board_state
        del ai
        gc.collect()
        if device == torch.device("cuda"):
            torch.cuda.empty_cache()

        with open('/content/drive/MyDrive/TorusGo/data.pickle', mode='wb') as fo:
            pickle.dump(data, fo)
        with open('/content/drive/MyDrive/TorusGo/kolog.pickle', mode='wb') as fo:
            pickle.dump(ko_point, fo)

        ai = Engine(board_state=next_board,current_player_color=player,ko_point=ko_point)

        if interval>=1:
            # 損失関数と最適化手法
            model = Custom2DNetSpatialOutput(
                input_shape_original_3d=input_shape,
                output_spatial_shape_2d=output_spatial_shape,
                resnet_layers=resnet_layers_config,
                num_classes_for_resnet=num_classes_for_resnet
            ).to(device)
            if os.path.isfile('/content/drive/MyDrive/TorusGo/model_weight_policy.pth'):
                model.load_state_dict(torch.load('/content/drive/MyDrive/TorusGo/model_weight_policy.pth', map_location=device))
            else:
                print("No policyweight train")

            criterion = nn.KLDivLoss(reduction='batchmean') # ログ確率間のKLダイバージェンスを計算
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            print("--- モデル訓練開始 ---")
            print(type(data[0][0]))
            train_model_policy(model, data[0]+data[1], criterion, optimizer, num_epochs=5, device=device)
            print("--- モデル訓練終了 ---")
            torch.save(model.state_dict(), "/content/drive/MyDrive/TorusGo/model_weight_policy.pth")
            interval = 0
            del model
            gc.collect()
            if device == torch.device("cuda"):
                torch.cuda.empty_cache()





    black_stones,white_stones = ai.root.count_stones()
    data[0].insert(0, 1 if black_stones > white_stones else -1)
    data[1].insert(0, -1 if black_stones > white_stones else 1)
    dataset.append(data[0])
    dataset.append(data[1])

    # Custom3DNetScalarOutputのインスタンス化
    model = Custom2DNetScalarOutput(
        input_shape=input_shape,
        output_spatial_shape=output_spatial_shape,
        resnet_layers=resnet_layers_config,
        num_classes_for_resnet=num_classes_for_resnet
    ).to(device)

    if os.path.isfile('/content/drive/MyDrive/TorusGo/model_weight_value.pth'):
        model.load_state_dict(torch.load('/content/drive/MyDrive/TorusGo/model_weight_value.pth', map_location=device))
    else:
        print("No valueweight train")


    # 損失関数と最適化手法
    criterion = nn.MSELoss() # 回帰タスクのためMSELossを使用
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("--- モデル訓練開始 ---")
    train_model_value(model, data[0], criterion, optimizer, num_epochs=5, device=device)
    train_model_value(model, data[1], criterion, optimizer, num_epochs=5, device=device)
    print("--- モデル訓練終了 ---")
    torch.save(model.state_dict(), "/content/drive/MyDrive/TorusGo/model_weight_value.pth")

with open('/content/drive/MyDrive/TorusGo/dataset.pickle', mode='wb') as fo:
    pickle.dump(dataset, fo)
