// server/game/GameLogic.js

class GoBoard {
    constructor(size, torusType = 'none') {
        this.size = size;
        this.board = Array(size).fill(null).map(() => Array(size).fill(0)); // 0:空, 1:黒, 2:白
        console.log("size:",size);
        //console.log("board:", this.board);
        this.torusType = torusType; // 'none', 'all', 'horizontal'
        this.koPoint = null; // [row, col]
        this.previousBoardState = null; // コウ判定のための前の盤面状態
    }

    // 座標変換ヘルパー
    _wrapCoord(coord) {
        return (coord % this.size + this.size) % this.size;
    }

    /**
     * 指定された座標の隣接点を取得（トーラス構造を考慮）
     * @param {number} r 行
     * @param {number} c 列
     * @returns {Array<[number, number]>} 隣接点の配列
     */
    getNeighbors(r, c) {
        const neighbors = [];
        const directions = [
            [-1, 0], // 上
            [1, 0],  // 下
            [0, -1], // 左
            [0, 1]   // 右
        ];

        for (const [dr, dc] of directions) {
            let nr = r + dr;
            let nc = c + dc;

            if (this.torusType === 'all') {
                nr = this._wrapCoord(nr);
                nc = this._wrapCoord(nc);
            } else if (this.torusType === 'horizontal') {
                nc = this._wrapCoord(nc);
                if (nr < 0 || nr >= this.size) continue; // 行はラップしない
            } else { // 'none'
                if (nr < 0 || nr >= this.size || nc < 0 || nc >= this.size) continue;
            }
            neighbors.push([nr, nc]);
        }
        return neighbors;
    }

    /**
     * 指定された座標の石と連結している同色の石の塊（グループ）を取得
     * @param {number} r 行
     * @param {number} c 列
     * @param {number} color 0:空, 1:黒, 2:白
     * @param {Set<string>} visited 訪問済み座標のセット（再帰用）
     * @returns {{group: Set<string>, liberties: Set<string>}} グループ内の石と呼吸点のセット
     */
    getGroupAndLiberties(r, c, color, visited = new Set()) {
        const group = new Set();
        const liberties = new Set();
        const stack = [[r, c]];

        visited.add(`${r},${c}`);

        while (stack.length > 0) {
            const [currR, currC] = stack.pop();
            group.add(`${currR},${currC}`);

            for (const [nr, nc] of this.getNeighbors(currR, currC)) {
                const neighborCoord = `${nr},${nc}`;
                if (!visited.has(neighborCoord)) {
                    visited.add(neighborCoord);
                    const neighborColor = this.board[nr][nc];

                    if (neighborColor === color) {
                        stack.push([nr, nc]);
                    } else if (neighborColor === 0) {
                        liberties.add(neighborCoord);
                    }
                }
            }
        }
        return { group, liberties };
    }

    /**
     * 石を置く処理と、石取りの実行
     * このメソッドは盤面を実際に変更します。
     * @param {number} r 行
     * @param {number} c 列
     * @param {number} color 1:黒, 2:白
     * @returns {Array<[number, number]> | null} 取り除かれた石の配列、自殺手の場合はnull
     */
    placeStoneAndCapture(r, c, color) {
        if (this.board[r][c] !== 0) {
            return []; // 既に石がある場合は取り除く石なし
        }

        console.log(`board:`,this.board);

        this.board[r][c] = color; // 石を仮置き

        const opponentColor = color === 1 ? 2 : 1;
        let removedStones = [];
        let anyOpponentStonesCaptured = false;

        // 相手の石が取られるかチェック
        for (const [nr, nc] of this.getNeighbors(r, c)) {
            if (this.board[nr][nc] === opponentColor) {
                const { group, liberties } = this.getGroupAndLiberties(nr, nc, opponentColor);
                if (liberties.size === 0) {
                    group.forEach(coordStr => {
                        const [rr, cc] = coordStr.split(',').map(Number);
                        removedStones.push([rr, cc]);
                        this.board[rr][cc] = 0; // 石を取り除く
                    });
                    anyOpponentStonesCaptured = true;
                }
            }
        }

        // 自殺手チェック (相手の石を取っていない場合のみ)
        if (!anyOpponentStonesCaptured) {
            const { liberties } = this.getGroupAndLiberties(r, c, color);
            if (liberties.size === 0) {
                // 自殺手なので、置いた石を元に戻す
                this.board[r][c] = 0;
                return null;
            }
        }
        return removedStones;
    }

    /**
     * 指定された手が合法かどうかを判定
     * このメソッドは盤面を一時的にコピーし、シミュレーションを行う。
     * 実際の盤面状態は変更しない。
     * @param {number} r 行
     * @param {number} c 列
     * @param {number} color 手番の色
     * @param {Array<Array<number>>} currentBoard 現在の盤面状態
     * @param {Array<Array<number>> | null} previousBoardState コウ判定のための直前の盤面状態
     * @param {Array<number> | null} koPoint コウ点
     * @returns {boolean} 有効な手であればtrue
     */
    isValidMove(r, c, color, currentBoard, previousBoardState, koPoint) {
        console.log("[isValidMove] checking for r,c:", r, c);
        // 既に石がある場所には打てない
        if (currentBoard[r][c] !== 0) {
          console.log("There is already a stone at", r, c);
          return false;
        }

        // コウ点への着手禁止
        if (koPoint && koPoint[0] === r && koPoint[1] === c) {
            console.log("Move is at ko point:", r, c);
            return false;
        }

        // 一時的に盤面をコピーしてシミュレート
        const tempBoard = new GoBoard(this.size, this.torusType);
        tempBoard.board = JSON.parse(JSON.stringify(currentBoard)); // 現在の盤面をコピー

        // シミュレーションで石を置いてみる
        const removedStones = tempBoard.placeStoneAndCapture(r, c, color);

        // 自殺手であれば無効
        if (removedStones === null) {
            console.log("Move is suicidal at", r, c);
            return false;
        }

        // コウ判定: 新しい盤面状態が直前の盤面状態と完全に同じになる場合、コウとみなす
        // ただし、相手の石を取った場合、かつ取った石が1つで、その場所がコウ点と一致する場合にのみ適用されるべき
        if (previousBoardState && removedStones.length === 1) {
            // シミュレーション後の盤面がpreviousBoardStateと一致するかチェック
            if (JSON.stringify(tempBoard.board) === JSON.stringify(previousBoardState)) {
                console.log("Move results in Ko re-capture at", r, c);
                return false;
            }
        }
        console.log("Move is valid at", r, c);
        return true;
    }

    /**
     * 終局時の盤上の石数を数える
     * @returns {{black: number, white: number}} 黒と白の石数
     */
    countStones() {
        let blackStones = 0;
        let whiteStones = 0;
        for (let r = 0; r < this.size; r++) {
            for (let c = 0; c < this.size; c++) {
                if (this.board[r][c] === 1) {
                    blackStones++;
                } else if (this.board[r][c] === 2) {
                    whiteStones++;
                }
            }
        }
        return { black: blackStones, white: whiteStones };
    }
}


class GameManager {
    constructor(gameId, player1Id, player2Id, player1Username, player2Username, boardSize, torusType, timeLimit, byoyomiTime, byoyomiCount) {
        this.gameId = gameId;
        this.player1 = {
            id: player1Id,
            username: player1Username,
            color: 1,
            remainingTime: timeLimit * 60, // 持ち時間（秒）
            byoyomiPeriods: byoyomiCount,
            currentByoyomiTime: (timeLimit === 0) ? byoyomiTime : 0 // 秒読み中の残り秒数（一時的な値）
        };
        this.player2 = {
            id: player2Id,
            username: player2Username,
            color: 2,
            remainingTime: timeLimit * 60, // 持ち時間（秒）
            byoyomiPeriods: byoyomiCount,
            currentByoyomiTime: (timeLimit === 0) ? byoyomiTime : 0 // 秒読み中の残り秒数（一時的な値）
        };
        this.currentTurn = 1; // 1:黒, 2:白
        console.log("boardSize:",boardSize);
        this.board = new GoBoard(boardSize, torusType);
        this.kifu = []; // 棋譜データ
        this.gameEnded = false;
        this.gameResult = null; // 'black_win', 'white_win', 'draw' など
        this.passCount = 0; // 連続パスの数
        this.timerInterval = null; // タイマーのインターバルID
        this.timeLimit = timeLimit; // 持ち時間（分）
        this.byoyomiTime = byoyomiTime; // 秒読み時間（秒）
        this.byoyomiCount = byoyomiCount; // 秒読み回数
        this.pendingEndGameRequest = null; // 終局申請中のプレイヤーID

        if (this.timeLimit === 0) {
            this.player1.currentByoyomiTime = this.byoyomiTime;
            this.player2.currentByoyomiTime = this.byoyomiTime;
        }
    }

    startGame() {
        this.gameEnded = false;
        //ゲーム開始時にlastMoveTimeをリセットし、すぐにタイマーを開始
        this.lastMoveTime = Date.now();
        this.startTimer();
    }

    startTimer(io = null) {
        if (this.timerInterval) clearInterval(this.timerInterval);

        this.timerInterval = setInterval(() => {
            if (this.gameEnded || this.pendingEndGameRequest) {
                this.stopTimer();
                return;
            }

            const currentPlayer = this.currentTurn === this.player1.color ? this.player1 : this.player2;
            const opponentPlayer = this.currentTurn === this.player1.color ? this.player2 : this.player1;

            // 持ち時間が残っている場合
            if (currentPlayer.remainingTime > 0) {
                currentPlayer.remainingTime--; // 1秒減らす
                if (currentPlayer.remainingTime <= 0) {
                    currentPlayer.remainingTime = 0; // 0以下にはしない
                    // 持ち時間切れ。秒読みシステムが設定されているか確認。
                    if (this.byoyomiCount > 0) { // 秒読みが設定されている場合
                        //持ち時間切れで秒読みに移行する際、byoyomiPeriodsは減らさない
                        currentPlayer.currentByoyomiTime = this.byoyomiTime; // 秒読み開始
                        console.log(`${currentPlayer.username} entered byoyomi. Periods left: ${currentPlayer.byoyomiPeriods}`);
                    } else {
                        // 持ち時間も秒読みも設定されていない場合、時間切れ
                        this.endGame('timeout', opponentPlayer.id, io);
                        this.stopTimer();
                        return;
                    }
                }
            } else {
                // 既に秒読み中の場合、または持ち時間が最初から0で秒読みに移行している場合
                // byoyomiPeriods が 0 の場合も、最後の秒読みとしてカウントダウンを続行
                if (currentPlayer.byoyomiPeriods >= 1) {
                    currentPlayer.currentByoyomiTime--; // 1秒減らす
                    if (currentPlayer.currentByoyomiTime <= 0) {
                        // 秒読み時間が切れた
                        if (currentPlayer.byoyomiPeriods > 1) {
                            currentPlayer.byoyomiPeriods--; // 秒読み回数を1減らす
                            currentPlayer.currentByoyomiTime = this.byoyomiTime; // 次の秒読み期間開始
                            console.log(`${currentPlayer.username} used a byoyomi period. Periods left: ${currentPlayer.byoyomiPeriods}`);
                        } else {
                            // 秒読み期間も使い切った
                            this.endGame('timeout', opponentPlayer.id, io);
                            this.stopTimer();
                            return;
                        }
                    }
                } else {
                    // このケースは本来到達しないはずだが、念のため
                    this.endGame('timeout', opponentPlayer.id, io);
                    this.stopTimer();
                    return;
                }
            }

            // UI更新のためのイベント発行
            if (io) {
                io.to(this.gameId).emit('update_game_state', this.getGameState());
            }
        }, 1000); // 1秒ごとに実行
    }

    stopTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
    }

    makeMove(r, c, playerId, io) {
        if (this.gameEnded || this.pendingEndGameRequest) {
            return { success: false, error: 'ゲームは終了しているか、終局申請中です' };
        }
        const currentPlayer = this.currentTurn === this.player1.color ? this.player1 : this.player2;
        if (currentPlayer.id !== playerId) {
            return { success: false, error: 'あなたの手番ではありません' };
        }

        // ここで isValidMove を呼び出して合法性をチェック
        const isValid = this.board.isValidMove(r, c, this.currentTurn, this.board.board, this.board.previousBoardState, this.board.koPoint);
        if (!isValid) {
            return { success: false, error: 'その場所には着手できません（既に石がある、自殺手、コウによる着手禁止）' };
        }

        this.board.previousBoardState = JSON.parse(JSON.stringify(this.board.board)); // コウ判定のため保存

        const removedStones = this.board.placeStoneAndCapture(r, c, this.currentTurn);

        // removedStonesがnullの場合はisValidMoveで弾かれているはずだが、念のため
        if (removedStones === null) {
            // これはisValidMoveがtrueを返した場合でも理論的には起こりうるが、設計上は発生しないはず
            this.board.previousBoardState = null; // 元に戻す
            return { success: false, error: '内部エラー: 自殺手と判定されました' };
        }

        // コウ点の設定
        // 1つだけ石を取り、かつその手で置いた石自体は単独で呼吸点を持つ場合
        if (removedStones.length === 1 &&
            this.board.getGroupAndLiberties(r, c, this.currentTurn).group.size === 1 &&
            this.board.getGroupAndLiberties(r, c, this.currentTurn).liberties.size === 1) {
            this.board.koPoint = removedStones[0];
        } else {
            this.board.koPoint = null;
        }

        this.kifu.push({
            turn: this.currentTurn,
            move: [r, c],
            removed: removedStones,
            board_before: this.board.previousBoardState,
            board_after: JSON.parse(JSON.stringify(this.board.board))
        });

        this.passCount = 0;
        this.currentTurn = this.currentTurn === 1 ? 2 : 1;
        //手を打ったら現在の秒読み時間をリセット（手番が渡るため）
        this.player1.currentByoyomiTime = this.byoyomiTime;
        this.player2.currentByoyomiTime = this.byoyomiTime;

        //持ち時間がある場合はcurrentByoyomiTimeをリセットしない
        if (this.player1.remainingTime > 0) {
            this.player1.currentByoyomiTime = 0; // 持ち時間中は秒読み時間を0に保つ
        }
        if (this.player2.remainingTime > 0) {
            this.player2.currentByoyomiTime = 0; // 持ち時間中は秒読み時間を0に保つ
        }

        //lastMoveTime を更新し、タイマーを再開
        this.lastMoveTime = Date.now();
        this.startTimer(io);

        return { success: true, removedStones, koPoint: this.board.koPoint, boardState: this.board.board, nextTurn: this.currentTurn };
    }

    passTurn(playerId, io) {
        if (this.gameEnded || this.pendingEndGameRequest) {
            return { success: false, error: 'ゲームは終了しているか、終局申請中です' };
        }
        const currentPlayer = this.currentTurn === this.player1.color ? this.player1 : this.player2;
        if (currentPlayer.id !== playerId) {
            return { success: false, error: 'あなたの手番ではありません' };
        }

        this.kifu.push({ turn: this.currentTurn, move: 'pass' });
        this.passCount++;
        this.board.previousBoardState = null; // パスしたのでコウ判定をリセット
        this.board.koPoint = null; // コウ点もリセット

        if (this.passCount >= 2) {
            this.endGame('pass_agreement', null, io);
            return { success: true, gameEnded: true, winner: this.gameResult };
        }

        this.currentTurn = this.currentTurn === 1 ? 2 : 1;
        //パスしたら現在の秒読み時間をリセット（手番が渡るため）
        this.player1.currentByoyomiTime = this.byoyomiTime;
        this.player2.currentByoyomiTime = this.byoyomiTime;

        //持ち時間がある場合はcurrentByoyomiTimeをリセットしない
        if (this.player1.remainingTime > 0) {
            this.player1.currentByoyomiTime = 0; // 持ち時間中は秒読み時間を0に保つ
        }
        if (this.player2.remainingTime > 0) {
            this.player2.currentByoyomiTime = 0; // 持ち時間中は秒読み時間を0に保つ
        }

        //lastMoveTime を更新し、タイマーを再開
        this.lastMoveTime = Date.now();
        this.startTimer(io);
        return { success: true, nextTurn: this.currentTurn };
    }

    resign(playerId, io) {
        if (this.gameEnded) return { success: false, error: 'ゲームは既に終了しています' };

        // 投了したプレイヤーを特定し、その相手を勝者とする
        const resigningPlayer = (this.player1.id === playerId) ? this.player1 : this.player2;
        const winnerPlayer = (this.player1.id === playerId) ? this.player2 : this.player1;

        if (resigningPlayer.id !== playerId) {
            return { success: false, error: 'あなたの手番ではありません' }; // 厳密には不要だが念のため
        }

        this.endGame('resignation', winnerPlayer.id, io);
        return { success: true, gameEnded: true, winner: this.gameResult };
    }

    requestEndGame(playerId, io) {
        if (this.gameEnded || this.pendingEndGameRequest) {
            return { success: false, error: 'ゲームは既に終了しているか、終局申請中です' };
        }

        this.kifu.push({ turn: this.currentTurn, move: 'end_game_request', requesterId: playerId });
        this.pendingEndGameRequest = playerId;
        this.stopTimer(); // 終局申請中はタイマー停止

        io.to(this.gameId).emit('receive_end_game_request', { requesterId: playerId });
        return { success: true };
    }

    respondToEndGame(playerId, agree, io) {
        if (!this.pendingEndGameRequest) return { success: false, error: '終局申請がありません' };
        if (this.pendingEndGameRequest === playerId) return { success: false, error: '自分自身の申請に応答できません' };

        if (agree) {
            this.endGame('agreement', null, io);
            return { success: true, gameEnded: true, winner: this.gameResult };
        } else {
            this.pendingEndGameRequest = null; // 申請をクリア
            this.startTimer(io); // タイマー再開
            this.kifu.push({ turn: this.currentTurn, move: 'end_game_rejected', respondentId: playerId });
            io.to(this.gameId).emit('end_game_rejected', { respondentId: playerId });
            return { success: true, resumed: true, msg: '終局申請が拒否され、対局が続行されます。' };
        }
    }

    async endGame(type, winnerId = null, io = null) {
        this.gameEnded = true;
        this.stopTimer();
        this.pendingEndGameRequest = null; // 終了時はクリア

        let winnerPlayerId;
        let loserPlayerId;
        let blackStones, whiteStones;

        if (type === 'timeout') {
            winnerPlayerId = winnerId;
            loserPlayerId = (winnerId === this.player1.id) ? this.player2.id : this.player1.id;
            this.gameResult = winnerPlayerId === this.player1.id ? `${this.player1.username}_win_timeout` : `${this.player2.username}_win_timeout`; // ★修正: 勝者のユーザー名を含む結果文字列
        } else if (type === 'resignation') {
            winnerPlayerId = winnerId;
            loserPlayerId = (winnerId === this.player1.id) ? this.player2.id : this.player1.id;
            this.gameResult = winnerPlayerId === this.player1.id ? `${this.player1.username}_win_resignation` : `${this.player2.username}_win_resignation`; // ★修正: 勝者のユーザー名を含む結果文字列
        } else if (type === 'pass_agreement' || type === 'agreement') {
            ({ black: blackStones, white: whiteStones } = this.board.countStones());
            if (blackStones > whiteStones) {
                winnerPlayerId = this.player1.id;
                loserPlayerId = this.player2.id;
                this.gameResult = `${this.player1.username}_win_agreement`; // ★修正: 勝者のユーザー名を含む結果文字列
            } else if (whiteStones > blackStones) {
                winnerPlayerId = this.player2.id;
                loserPlayerId = this.player1.id;
                this.gameResult = `${this.player2.username}_win_agreement`; // ★修正: 勝者のユーザー名を含む結果文字列
            } else {
                winnerPlayerId = null; // 引き分け
                loserPlayerId = null;
                this.gameResult = 'draw';
            }
        }
        //プレイヤーが接続を切った場合の処理（対局中の場合のみ）
        else if (type === 'player_disconnected') {
            winnerPlayerId = winnerId;
            loserPlayerId = (winnerId === this.player1.id) ? this.player2.id : this.player1.id;
            this.gameResult = winnerPlayerId === this.player1.id ? `${this.player1.username}_win_by_disconnect` : `${this.player2.username}_win_by_disconnect`; // ★追加: 切断による勝利
        }
        //AI戦で人間プレイヤーが離れた場合の処理（対局中の場合のみ）
        else if (type === 'player_left_ai_game') {
            winnerPlayerId = 'ai-player-id'; // AIが勝者
            loserPlayerId = winnerId; // 離れた人間プレイヤーが敗者
            this.gameResult = 'AI_win_by_player_left'; // ★追加: AI戦でのプレイヤー離脱によるAIの勝利
        }


        // DBのUserモデルを更新
        const User = require('../models/User'); // ここでUserモデルをロード
        try {
            if (winnerPlayerId && winnerPlayerId !== 'ai-player-id') { // AIプレイヤーはDB更新しない
                await User.findByIdAndUpdate(winnerPlayerId, { $inc: { 'score.wins': 1 }, is_playing: false });
            }
            if (loserPlayerId && loserPlayerId !== 'ai-player-id') { // AIプレイヤーはDB更新しない
                await User.findByIdAndUpdate(loserPlayerId, { $inc: { 'score.losses': 1 }, is_playing: false });
            }
            // 引き分けの場合 (AI戦以外)
            if (type === 'pass_agreement' || type === 'agreement' && this.gameResult === 'draw' && this.player2.id !== 'ai-player-id') {
                 await User.findByIdAndUpdate(this.player1.id, { is_playing: false });
                 await User.findByIdAndUpdate(this.player2.id, { is_playing: false });
            }
            // AI戦で人間プレイヤーが離れた場合、人間プレイヤーのis_playingをfalseにする
            if (type === 'player_left_ai_game' && winnerId !== 'ai-player-id') {
                await User.findByIdAndUpdate(winnerId, { is_playing: false });
            }
            //終局後に退室したプレイヤーのis_playingをfalseにする（既に終わっているゲームの場合）
            // このロジックはhandlers.jsのdisconnectイベントでis_playingをfalseにする処理と連携
            // ここではゲームが終了したときに両プレイヤーのis_playingをfalseにする
            if (this.player1.id !== 'ai-player-id') {
                await User.findByIdAndUpdate(this.player1.id, { is_playing: false });
            }
            if (this.player2.id !== 'ai-player-id') {
                await User.findByIdAndUpdate(this.player2.id, { is_playing: false });
            }

        } catch (error) {
            console.error('Failed to update user scores:', error);
        }

        if (io) {
            // game_over イベントの data に winnerUsername も追加
            const winnerUsername = winnerPlayerId === this.player1.id ? this.player1.username :
                                   winnerPlayerId === this.player2.id ? this.player2.username :
                                   (winnerPlayerId === 'ai-player-id' ? 'AI' : null);

            io.to(this.gameId).emit('game_over', {
                result: this.gameResult,
                winnerId: winnerPlayerId,
                winnerUsername: winnerUsername,
                kifu: this.kifu
            });
        }
        console.log(`Game ${this.gameId} ended. Result: ${this.gameResult}`);
    }

    getGameState() {
        return {
            gameId: this.gameId,
            board: this.board.board,
            turn: this.currentTurn,
            player1: {
                id: this.player1.id,
                username: this.player1.username,
                color: this.player1.color,
                remainingTime: this.player1.remainingTime,
                byoyomiPeriods: this.player1.byoyomiPeriods,
                currentByoyomiTime: this.player1.currentByoyomiTime // ★追加
            },
            player2: {
                id: this.player2.id,
                username: this.player2.username,
                color: this.player2.color,
                remainingTime: this.player2.remainingTime,
                byoyomiPeriods: this.player2.byoyomiPeriods,
                currentByoyomiTime: this.player2.currentByoyomiTime // ★追加
            },
            koPoint: this.board.koPoint,
            gameEnded: this.gameEnded,
            gameResult: this.gameResult,
            pendingEndGameRequest: this.pendingEndGameRequest,
            boardSize: this.board.size,
            torusType: this.board.torusType,
            kifu: this.kifu,
            //持ち時間と秒読みの初期設定をGameStateに含める
            timeLimit: this.timeLimit,
            byoyomiTime: this.byoyomiTime,
            byoyomiCount: this.byoyomiCount
        };
    }
}

module.exports = { GoBoard, GameManager };
