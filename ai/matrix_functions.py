def rotate_board(boards):
    """
    トーラス構造に則って碁盤を回転させる
    """
    rotated_boards = [[] for i in range(len(boards))]
    col_rotated = [0 for i in range(len(boards))]
    row_rotated = [0 for i in range(len(boards))]
    for i in range(6):
        for j in range(6):
            for x,board in zip(range(len(boards)),boards):
                col_rotated[x] = [row[:] for row in board]
                col_rotated[x] = col_rotated[x][i:]+col_rotated[x][:i]
                #rotated_boards[x].append([list(q) for q in list(zip(*col_rotated[x]))])
                row_rotated[x] = [row[:] for row in [list(q) for q in list(zip(*col_rotated[x]))]]
                row_rotated[x] = row_rotated[x][j:]+row_rotated[x][:j]
                rotated_boards[x].append([list(q) for q in list(zip(*row_rotated[x]))])
    return rotated_boards


def shape_board2tensor(prev_cur_boards,turn):
    """
    盤面データをニューラルネット入力用のテンソルに変換
    """
    import copy
    blacks = [[[0 if (value==2) or (value==0) else 1 for value in row] for row in rowcol] for rowcol in prev_cur_boards]
    whites = [[[0 if (value==1) or (value==0) else 1 for value in row] for row in rowcol] for rowcol in prev_cur_boards]
    #print("blacks",get_array_shape(blacks))
    tensor2input = []
    b = []
    w = []
    for board in zip(blacks,whites):
        b_board = [row[:] for row in board[0]] # ディープコピー
        w_board = [row[:] for row in board[1]]
        rotated = rotate_board([b_board,w_board])
        b.append(rotated[0])
        w.append(rotated[1])

    tensor2input = torch.tensor([b+w+[full_array(b[0],1 if turn==1 else -1)]+[full_array(b[0],komi/3)]])
    return tensor2input.float().to(device)

#以下一般的な配列操作関数　引数の形状に注意

def range_ndim(*dimensions_args):
    """
    指定された次元と範囲に基づいて多次元の座標を生成するイテレータを返す

　　引数がややこしい　メモ
    Args:
        *dimensions_args: 各次元の範囲を指定する引数
                          各引数の形式:
                          - stop (int): (0, stop) の範囲
                          - (start, stop) (tuple): start から stop-1 までの範囲
                          - (start, stop, step) (tuple): start から stop-1 まで step 間隔の範囲

    Yields:
        tuple: 各次元の座標を表すタプル
    """
    # 各次元のrangeオブジェクトを事前に準備
    ranges = []
    for arg in dimensions_args:
        if isinstance(arg, int):
            ranges.append(range(arg))
        elif isinstance(arg, tuple):
            ranges.append(range(*arg))
        else:
            raise TypeError("dimensions_argsの各要素はintまたはタプル")

    def _generate_coords(current_dimension_index, current_coords):
        current_range = ranges[current_dimension_index]

        # 現在の次元の各値をループ
        for value in current_range:
            new_coords = current_coords + (value,)

            # 最終次元に到達した場合
            if current_dimension_index == len(ranges) - 1:
                yield new_coords # 完成した座標タプルをyield
            else:
                # 最終次元でなければ、次の次元に進んで再帰呼び出し
                yield from _generate_coords(current_dimension_index + 1, new_coords)

    # range_ndimに引数が与えられない場合は、空のジェネレータを返す
    if not ranges:
        return

    yield from _generate_coords(0, ())

def get_indices(arr, target_value, current_indices=None):
    """
    リスト（何次元でも）から指定された値の全ての添え字を返す

    Args:
        arr (list): 多次元リスト
        target_value: 指定する値
        current_indices (list, optional): 現在までのインデックスのパス
                                         再帰に使う

    Returns:
        list: 指定された値が見つかった全ての添え字のリスト
              形: [[0, 1], [2, 0]]
    """
    if current_indices is None:
        current_indices = []

    results = []

    if isinstance(arr, list):
        for i, item in enumerate(arr):
            results.extend(get_indices(item, target_value, current_indices + [i]))
    else:
        if arr == target_value:
            results.append(current_indices)

    return results

def flatten_list_scalars_generator(nested_list):
    """
    ネストされたリストを平たくし、全てのスカラーをジェネレータとしてyield
    """
    for item in nested_list:
        if isinstance(item, list):
            yield from flatten_list_scalars_generator(item)
        else:
            yield item
"""
def get_coords_of_non_value(data: list | np.ndarray, exclude_values: list | tuple) -> tuple:
    #指定した複数の値以外の要素の座標を返す
    #引数はNumPy配列でもPythonの標準リストでも可

    #Args:
        #data (list | np.ndarray): 任意の形状のNumPy配列または入れ子になったPythonリスト。
        #exclude_values (list | tuple): 除外したい値のリストまたはタプル。
    #Returns:
        #tuple: 指定した値以外の要素の座標を示すタプル。
               #各要素は、対応する次元のインデックスのNumPy配列です。
               #入力が空の場合や該当する値がない場合は、空のNumPy配列を含むタプルを返します。


    # 入力がリストの場合、NumPy配列に変換
    if isinstance(data, list):
        if not data:
            return tuple(np.array([]) for _ in range(0)) # 空のリストが渡された場合

        try:
            arr = np.array(data)
        except ValueError as e:
            print(f"警告: 入力リストの形状が不均一　NumPy配列への変換に失敗: {e}")
            raise
    elif isinstance(data, np.ndarray):
        arr = data
    else:
        raise TypeError("NumPy配列またはPythonリストを入力")

    # 初期マスクをTrueで初期化（すべての要素を「除外しない」と仮定）
    # arr.shape と np.ones の組み合わせで、元の配列と同じ形状のTrueの配列を作成
    mask = np.ones(arr.shape, dtype=bool)

    # 除外したい値ごとにマスクを更新
    for val in exclude_values:
        mask = mask & (arr != val) # 論理ANDでマスクを結合

    # マスクがTrueである要素の座標を取得
    coordinates = np.transpose(np.where(mask)).tolist()

    return coordinates
"""
from typing import Union, List, Tuple
def get_coords_of_non_value(data: Union[List, np.ndarray], exclude_values: Union[List, Tuple]) -> List[List[int]]:
    """
    指定した複数の値以外の要素の座標を返す
    引数はNumPy配列でもPythonの標準リストでも可能

    Args:
        data (list | np.ndarray): 任意の形状のNumPy配列または入れ子になった配列
        exclude_values (list | tuple): 除外したい値の配列またはタプル

    Returns:
        list[list[int]]: 指定した値以外の要素の座標を示すリストのリスト
                          各内側のリストは、対応する要素の次元ごとのインデックス
                          入力が空の場合や該当する値がない場合は空のリストを返す
    """
    # 入力がリストの場合、NumPy配列に変換
    if isinstance(data, list):
        if not data:
            return []

        try:
            arr = np.array(data)
        except ValueError as e:
            # 不均一なリストがNumPy配列に変換できない場合エラーを表示
            raise ValueError(f"警告: 入力リストの形状が不均一　NumPy配列への変換に失敗: {e}")
    elif isinstance(data, np.ndarray):
        arr = data
    else:
        raise TypeError("NumPy配列またはPythonリストを入力")

    if arr.size == 0:
        return []

    # マスク　Trueで初期化
    mask = np.ones(arr.shape, dtype=bool)

    # 除外したい値ごとにマスクを更新
    mask = ~np.isin(arr, exclude_values)

    # マスクがTrueである要素の座標を取得
    coordinates = np.transpose(np.where(mask)).tolist()

    return coordinates

def max_in_tensor(nested_array):
    """
    ネストされた配列内の個々のスカラーから最大値を求める
    """
    # ジェネレータ式でスカラー要素を順次生成
    flat_scalars_gen = flatten_list_scalars_generator(nested_array)

    try:
        return max(flat_scalars_gen)
    except ValueError:
        raise ValueError("配列にスカラー要素なし")

def max_in_tensor_generator(nested_array):
    """
    ネストされた配列内の個々のスカラー要素を最大値から順に生成


    Args:
        nested_array: スカラー値を抽出したいネストされた配列

    Yields:
        int or float: 配列内のスカラー値を降順に一つずつ生成
    """
    #平たくする
    flat_scalars_gen = flatten_list_scalars_generator(nested_array)

    # ジェネレータから全ての値を取り出し、リストに変換して並び替える
    all_scalars = list(flat_scalars_gen)

    if not all_scalars:
        # スカラー要素が一つもない場合は何もyieldしない
        return

    # リストを降順に並び替え
    all_scalars.sort(reverse=True)

    # ソートされた要素を一つずつyieldする
    for scalar in all_scalars:
        yield scalar

def check_and_count_values_at_coordinates(
    arr_data: list | np.ndarray,
    coords_data: list | np.ndarray,
    value_a: any,
    value_b: any
) -> tuple[bool, int]:
    """
    配列/リスト arr_data の、coords_data に含まれる座標の値が
    全て a と等しいかを確認し、同時に b である数を数える

    Args:
        arr_data (list かnp.ndarray): 元のNumPy配列または入れ子配列
        coords_data (list かnp.ndarray): 確認したい座標が格納されたNumPy配列または入れ子はいれす
                                          形式は [[dim1_idx1, dim2_idx1, ...], [dim1_idx2, dim2_idx2, ...], ...]
        value_a (any): 全ての値が等しいか確認するための比較対象の値
        value_b (any): 数を数えるための比較対象の値

    Returns:
        tuple[bool, int]:
            bool: 指定した座標の全ての値が a と等しければ True
            int: 指定した座標の値の中で b と等しい値の数
    """

    if isinstance(arr_data, list):
        if not arr_data:
            return True, 0
        try:
            arr_a = np.array(arr_data)
        except ValueError as e:
            print(f"警告: 入力リストの形状が不均一　NumPy配列への変換に失敗: {e}")
            raise
    elif isinstance(arr_data, np.ndarray):
        arr_a = arr_data
    else:
        raise TypeError("引数はNumPy配列またはPythonリスト")


    if isinstance(coords_data, list):
        if not coords_data:
            return True, 0
        coords_b = np.array(coords_data)
    elif isinstance(coords_data, np.ndarray):
        coords_b = coords_data
    else:
        raise TypeError("NumPy配列またはPythonリストを入力")

    if coords_b.size == 0:
        return True, 0

    # 座標配列の形状をチェックし添え字を抽出
    if coords_b.ndim == 2 and arr_a.ndim == coords_b.shape[1]:
        indices = tuple(np.transpose(coords_b))
    elif coords_b.ndim == 1 and arr_a.ndim == 1:
        indices = coords_b
    else:
        raise ValueError("coords_bの形状がarr_aの次元数と不一致")

    # 指定された座標の値を抽出
    values_at_coords = arr_a[indices]

    #全ての値が value_a と等しいかを確認
    all_are_a = np.all(values_at_coords == value_a)

    #value_b である数を数える
    count_of_b = np.sum(values_at_coords == value_b)

    return all_are_a, int(count_of_b)


def get_coords_over_threshold(arr, threshold_value, current_coordinates=None):
    """
    入れ子配列から指定された閾値以上の値を持つ全ての座標（添え字）を集める

    Args:
        arr (list): 探索対象のネストされたリスト
        threshold_value: 比較する閾値
        current_coordinates (list, optional): 現在までの座標のパス
                                             再帰で使う
    Returns:
        list: 指定された条件を満たす値が見つかった全ての座標のリスト
              形: [[0, 1], [2, 0]]
    """
    if current_coordinates is None:
        current_coordinates = []

    results = []

    if isinstance(arr, list):
        for i, item in enumerate(arr):
            # current_coordinatesのコピーを渡して各パスを独立にする
            results.extend(get_coords_over_threshold(item, threshold_value, current_coordinates + [i]))
    else:
        if arr >= threshold_value:
            results.append(current_coordinates)

    return results

def get_array_shape(arr):
    """
    多次元リスト（配列）の形状をタプルで返す
    （均一専用）

    Args:
        arr: 形状を調べたい多次元リスト

    Returns:
        各次元の長さを表すタプル
    """
    if not isinstance(arr, list):
        return ()

    if not arr:
        return (0,)

    # 現在の次元の長さを取得
    current_dimension_length = len(arr)

    # 次の次元の形状を再帰的に取得
    # 配列が均一であると仮定
    try:
        next_dimension_shape = get_array_shape(arr[0])
    except IndexError:
        return (current_dimension_length,)
    except Exception as e:
        # 均一でない配列の場合ここでエラーが発生する可能性がある
        raise ValueError("不均一な形状") from e


    return (current_dimension_length,) + next_dimension_shape

def full_array(original_array, fill_value):
    """
    to create an array with the shape of original_array filled with fill_value

    Args:
        original_array: 形状をコピーしたい元のネストされたリスト
        fill_value: 新しい配列の各要素に設定する任意の値

    Returns:
        元の配列と同じ形状で、全ての要素がfill_valueである新しいリスト
    """
    if isinstance(original_array, list):
        # 要素がリストであれば、そのリストの各要素に対して再帰的に関数を呼び出す
        return [full_array(item, fill_value) for item in original_array]
    else:
        # 要素がリストでなければ、指定されたfill_valueで置き換える
        return fill_value


"""
def shape_board2tensor(prev_cur_boards,turn):
    import copy
    #boards = [row[:] for row in prev_boards.append(curr_board)]
    boards = copy.deepcopy(prev_cur_boards)
    #blacks = [[[0 if value==2 else value for value in row] for row in rowcol] for rowcol in boards]
    #whites = [[[0 if value==1 else value for value in row] for row in rowcol] for rowcol in boards]
    tensor2input = []
    b = []
    for board in boards:
        rotated = rotate_board([board])
        b.append(rotated[0])

    tensor2input = torch.tensor([b+[full_array(b[0],turn)]])
    return tensor2input.float().to(device)
"""
