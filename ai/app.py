# ai/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# go_engine.py があるディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from go_engine import GoEngine # AIの思考エンジンをインポート

app = Flask(__name__)
CORS(app) # CORSを有効にする

# ゲームのインスタンスを保持する辞書
game_engines = {}

@app.route('/ai/make_move', methods=['POST'])
def make_ai_move():
    data = request.json
    game_id = data.get('gameId')
    board_state = data.get('boardState')
    board_size = data.get('boardSize')
    torus_type = data.get('torusType')
    current_turn = data.get('currentTurn') # AIが打つ番の色 (1:黒, 2:白)
    ko_point = data.get('koPoint')

    if not all([game_id, board_state, board_size, torus_type, current_turn is not None]):
        return jsonify({"error": "Missing required game data"}), 400

    if game_id not in game_engines:
        game_engines[game_id] = GoEngine(board_size, torus_type)

    ai_engine = game_engines[game_id]
    ai_engine.set_board_state(board_state, ko_point) # 最新の盤面状態をAIに反映

    print(f"AI for game {game_id} is thinking... Player Color: {current_turn}")
    #best_move_row, best_move_col = ai_engine.find_best_move(current_turn)
    best_move_row, best_move_col = ai_engine.find_best_move(current_turn,game_id)

    if best_move_row is None and best_move_col is None:
        return jsonify({"move": "pass", "message": "AI decided to pass"}), 200

    print(f"AI move for game {game_id}: ({best_move_row}, {best_move_col})")
    return jsonify({"move": [best_move_row, best_move_col]}), 200

if __name__ == '__main__':
    # Flaskのデフォルトポートは5000なので、競合を避けるために5001に設定
    app.run(host='0.0.0.0', port=5001, debug=True)
