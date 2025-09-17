// client/src/pages/GameScreen.js

import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import GoBoardComponent,{ BOARD_AREA_SIZE, BOARD_PADDING } from '../components/GoBoardComponent';
import EndGameModal from '../components/EndGameModal';
import './GameScreen.css';

function GameScreen({ currentUser, socket }) {
    const { gameId } = useParams();
    const navigate = useNavigate();

    const [gameState, setGameState] = useState(null);
    const [mainBoardOffset, setMainBoardOffset] = useState({ x: 0, y: 0 });
    const [smallBoardOffset, setSmallBoardOffset] = useState({ x: 0, y: 0 });
    const [showEndGameConfirm, setShowEndGameConfirm] = useState(false);
    const [showIncomingEndGameRequest, setShowIncomingEndGameRequest] = useState(false);
    const [isGameEnded, setIsGameEnded] = useState(false);
    const [gameResult, setGameResult] = useState(null);
    const [kifuData, setKifuData] = useState(null); // 棋譜データ
    const [endGameReason, setEndGameReason] = useState(null);


    const player1Info = gameState?.player1;
    const player2Info = gameState?.player2;
    // 自分と相手の情報を正しく設定
    const myPlayerInfo = (player1Info && player1Info.id === currentUser.id) ? player1Info : player2Info;
    const opponentPlayerInfo = (player1Info && player1Info.id === currentUser.id) ? player2Info : player1Info;

    const myColor = myPlayerInfo?.color;
    const isMyTurn = gameState?.turn === myColor;

    useEffect(() => {
        // WebSocket接続状態のログ
        const handleConnect = () => console.log(`[GameScreen] Socket CONNECTED. socket.id: ${socket.id}`);
        const handleDisconnect = () => console.log(`[GameScreen] Socket DISCONNECTED. socket.id: ${socket.id}`);

        socket.on('connect', handleConnect);
        socket.on('disconnect', handleDisconnect);
        // ゲーム状態の初期リクエスト
        socket.emit('request_game_state', gameId);

        socket.on('update_game_state', (data) => {
            console.log('Received game state update:', data);
            setGameState(data);
            if (data.gameEnded) {
                setIsGameEnded(true);
                setGameResult(data.gameResult);
                setKifuData(data.kifu); // ゲーム終了時に棋譜データをセット
            }
            // 終局申請中の状態を反映
            if (data.pendingEndGameRequest && data.pendingEndGameRequest !== currentUser.id) {
                setShowIncomingEndGameRequest(true);
            } else {
                setShowIncomingEndGameRequest(false);
            }
        });

        socket.on('game_over', (data) => {
            setIsGameEnded(true);
            setGameResult(data.result);
            setKifuData(data.kifu); // サーバーから来た棋譜データを使用
            setGameState(prev => ({ ...prev, gameEnded: true, gameResult: data.result, kifu: data.kifu }));

            let message = '';
            if (data.result.includes('win_by_disconnect')) {
                // 対局中の切断のみメッセージを出す
                const disconnectedPlayerUsername = data.winnerId === myPlayerInfo.id ? opponentPlayerInfo.username : myPlayerInfo.username;
                message = `Game ended. ${disconnectedPlayerUsername} has been disconnected. ${data.winnerUsername} wins.`;
            } else if (data.result.includes('AI_win_by_player_left')) {
                message = `Player left the room`;
            } else if (data.result.includes('_win')) { // resignation, timeout, agreement, etc.
                message = `Winner: ${data.winnerUsername}!`;
            } else if (data.result === 'draw') {
                message = `Draw!`;
            } else {
                message = `Result: ${data.result}`;
            }
            alert(message);
            setEndGameReason(data.result); // 終局理由をステートに保存
            //ここでnavigateせず、終局画面で退室ボタンを押すとロビーに戻るようにする
        });

        socket.on('receive_end_game_request', (data) => {
            if (data.requesterId !== currentUser.id) {
                setShowIncomingEndGameRequest(true);
            }
        });

        socket.on('end_game_rejected', () => {
            setShowEndGameConfirm(false);
            setShowIncomingEndGameRequest(false);
            alert('End Game Request was declined, play will continue');
        });

        socket.on('move_failed', (data) => {
            alert(`Cannot put the point: ${data.message}`);
        });

        socket.on('action_failed', (data) => {
            alert(`Action failed: ${data.message}`);
        });

        socket.on('game_not_found', () => {
            // ゲームが既に終了している場合はロビーに戻らない
            if (!isGameEnded) {
                alert('Couldn’t find the game, returning to the lobby');
                navigate('/lobby');
            }
        });

        return () => {
            socket.off('connect', handleConnect);
            socket.off('disconnect', handleDisconnect);
            socket.off('update_game_state');
            socket.off('game_over');
            socket.off('receive_end_game_request');
            socket.off('end_game_rejected');
            socket.off('move_failed');
            socket.off('action_failed');
            socket.off('game_not_found');
        };
    }, [gameId, socket, currentUser.id, navigate, myPlayerInfo, opponentPlayerInfo, isGameEnded]); // ★修正: 依存配列にisGameEndedを追加


    const handlePlaceStone = (row, col) => {
        if (!isMyTurn || isGameEnded || gameState?.pendingEndGameRequest) return;
        // サーバーサイドで合法性チェックが行われるため、クライアント側では単純にemitする
        socket.emit('make_move', { gameId, userId: currentUser.id, row, col });
    };

    const handlePass = () => {
        if (!isMyTurn || isGameEnded || gameState?.pendingEndGameRequest) return;
        socket.emit('pass_turn', { gameId, userId: currentUser.id });
    };

    const handleResign = () => {
        if (isGameEnded) return;
        if (window.confirm('Are you sure to resign？')) {
            socket.emit('resign', { gameId, userId: currentUser.id });
        }
    };

    const handleRequestEndGame = () => {
        if (isGameEnded || gameState?.pendingEndGameRequest) return;
        setShowEndGameConfirm(true);
    };

    const confirmEndGameRequest = () => {
        socket.emit('request_end_game', { gameId, userId: currentUser.id });
        setShowEndGameConfirm(false);
    };

    const handleRespondToEndGameRequest = (agree) => {
        socket.emit('respond_to_end_game', { gameId, userId: currentUser.id, agree });
        setShowIncomingEndGameRequest(false);
    };

    const handleReturnToLobby = useCallback(() => {
        console.log("[GameScreen] Returning to lobby.");
        if (currentUser && currentUser.id) {
            console.log(`[GameScreen] Emitting set_user_id for currentUser.id: ${currentUser.id}, current socket.id: ${socket.id}`);
            socket.emit('set_user_id', currentUser.id);
            socket.emit('player_leaving_game', { gameId, userId: currentUser.id });
        } else {
            console.log("[GameScreen] currentUser or currentUser.id is missing, cannot emit set_user_id.");
        }
        navigate('/lobby');
    }, [navigate, socket, currentUser, gameId]);

    const handleDownloadKifu = () => {
        if (!isGameEnded || !kifuData) {
            alert('The game haven’t ended or No kifu data');
            return;
        }
        const kifuJson = JSON.stringify(kifuData, null, 2);
        const filename = `go_game_${gameId}_kifu.json`;
        const blob = new Blob([kifuJson], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    const calculateCellSize = useCallback((isSmallBoard, currentBoardSize) => {
        // GoBoardComponentのCELL_SIZE計算ロジックと同期
        //return isSmallBoard ? (BOARD_AREA_SIZE - BOARD_PADDING * 2) / ((currentBoardSize - 1) * 2) : (BOARD_AREA_SIZE - BOARD_PADDING * 2) / (currentBoardSize - 1);
        return isSmallBoard ? (BOARD_AREA_SIZE - BOARD_AREA_SIZE/10) / ((currentBoardSize - 1) * 2) : (BOARD_AREA_SIZE - BOARD_AREA_SIZE/10) / (currentBoardSize - 1);
    }, []);


    const handleBoardMove = useCallback((type, direction, reverse = false) => {
        const currentBoardSize = gameState?.boardSize;
        if (!currentBoardSize) return;

        const cellSize = calculateCellSize(false, currentBoardSize); // メインボード用
        const step = direction === 'single' ? 1 : 5;
        //const moveAmount = step * cellSize * (reverse ? 1 : -1);
        const moveAmount = step * (reverse ? 1 : -1);

        if (type === 'row') {
            setMainBoardOffset(prev => ({ ...prev, y: prev.y + moveAmount }));
        } else if (type === 'col') {
            setMainBoardOffset(prev => ({ ...prev, x: prev.x + moveAmount }));
        }
    }, [gameState?.boardSize, calculateCellSize]);

    const handleSmallBoardMove = useCallback((type, direction, reverse = false) => {
        const currentBoardSize = gameState?.boardSize;
        if (!currentBoardSize) return;

        const cellSize = calculateCellSize(true, currentBoardSize); // 小型ボード用
        const step = direction === 'single' ? 1 : 5;
        //const moveAmount = step * cellSize * (reverse ? 1 : -1);
        const moveAmount = step * (reverse ? 1 : -1);

        if (type === 'row') {
            setSmallBoardOffset(prev => ({ ...prev, y: prev.y + moveAmount }));
        } else if (type === 'col') {
            setSmallBoardOffset(prev => ({ ...prev, x: prev.x + moveAmount }));
        }
    }, [gameState?.boardSize, calculateCellSize]);

    if (!gameState) {
        return <div className="game-screen-container">Loading the game board...</div>;
    }

    /*
    // 時間をMM:SS形式にフォーマットするヘルパー関数
    const formatTime = useCallback((totalSeconds) => {
        if (totalSeconds <= 0) return '00:00';
        const minutes = Math.floor(totalSeconds / 60);
        const seconds = Math.floor(totalSeconds % 60); // 0.1秒単位以下は表示しないため、切り捨てる
        return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    }, []);
    */
    // 時間表示のフォーマット
    const formatTime = (seconds) => {
        if (seconds < 0) return '0分 00'; // マイナス表示はしない
        const minutes = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${minutes}m ${secs.toString().padStart(2, '0')}`;
        //return `${minutes}:${secs}`;
        return seconds;
    };


    // 秒読み時間の表示を調整する関数 (byoyomiCount は初期設定値、byoyomiPeriods は残り回数)
    const formatByoyomiTime = (byoyomiSeconds, byoyomiPeriods, totalByoyomiCount) => {
        if (byoyomiPeriods > 0) {
            return `${byoyomiSeconds.toString().padStart(2, '0')}s / ${byoyomiPeriods}times`;
        } else if (byoyomiPeriods === 0 && totalByoyomiCount > 0) {
            // byoyomiPeriodsが0だが、元々秒読みが設定されていた場合（最後の1回）
            return `${byoyomiSeconds.toString().padStart(2, '0')}s (last)`;
        } else {
            return 'Time over'; // 秒読みも使い切った場合、または秒読み設定がない場合
        }
    };

    // ゲーム結果メッセージの生成
    const getGameResultMessage = () => {
        if (!gameResult) return '';

        //勝敗文字列を解析してメッセージを生成
        const [winnerPrefix, winType] = gameResult.split('_win');
        let winnerName = null;
        if (winnerPrefix === player1Info.username) {
            winnerName = player1Info.username;
        } else if (winnerPrefix === player2Info.username) {
            winnerName = player2Info.username;
        } else if (winnerPrefix === 'AI') {
            winnerName = 'AI';
        }

        if (gameResult.includes('win_by_disconnect')) {
            const disconnectedPlayer = (winnerName === myPlayerInfo.username) ? opponentPlayerInfo.username : myPlayerInfo.username;
            return `${disconnectedPlayer} has been disconnected, ${winnerName} wins!`;
        } else if (gameResult.includes('AI_win_by_player_left')) {
            return 'Player left the room';
        } else if (gameResult.includes('_win_timeout')) {
            const loserName = (winnerName === player1Info.username) ? player2Info.username : player1Info.username;
            return `${loserName} ran out of time, ${winnerName} wins!`;
        } else if (gameResult.includes('_win_resignation')) {
            const loserName = (winnerName === player1Info.username) ? player2Info.username : player1Info.username;
            return `${loserName} resigned, ${winnerName} wins!`;
        } else if (gameResult.includes('_win_agreement')) {
            return `${winnerName} wins`;
        } else if (gameResult === 'draw') {
            return 'Draw!';
        }
        return `Result: ${gameResult}`;
    };


    return (
      <div className="game-screen-container">
          <header className="game-header">
              {isGameEnded && <div className="game-over-message">{getGameResultMessage()}</div>}
          </header>

          <div className="game-content">
              <div className="main-board-area">
                  <GoBoardComponent
                      boardState={gameState.board}
                      boardSize={gameState.boardSize}
                      isSmall={false}
                      torusType={gameState.torusType}
                      onPlaceStone={handlePlaceStone}
                      isMyTurn={isMyTurn}
                      offset={mainBoardOffset}
                      isClickable={true}
                  />
                  <div className="board-move-buttons">
                      <button onClick={() => handleBoardMove('col', 'single', false)}>←1</button>
                      <button onClick={() => handleBoardMove('col', 'five', false)}>←5</button>
                      <button onClick={() => handleBoardMove('col', 'single', true)}>→1</button>
                      <button onClick={() => handleBoardMove('col', 'five', true)}>→5</button>
                      <button onClick={() => handleBoardMove('row', 'single', false)} disabled={gameState.torusType === 'horizontal'}>↑1</button>
                      <button onClick={() => handleBoardMove('row', 'five', false)} disabled={gameState.torusType === 'horizontal'}>↑5</button>
                      <button onClick={() => handleBoardMove('row', 'single', true)} disabled={gameState.torusType === 'horizontal'}>↓1</button>
                      <button onClick={() => handleBoardMove('row', 'five', true)} disabled={gameState.torusType === 'horizontal'}>↓5</button>
                  </div>
                  <div className="game-controls">
                      <button onClick={handlePass} disabled={!isMyTurn || isGameEnded || gameState?.pendingEndGameRequest}>パス</button>
                      <button onClick={handleResign} disabled={isGameEnded}>投了</button>
                      <button onClick={handleRequestEndGame} disabled={isGameEnded || gameState?.pendingEndGameRequest}>終局申請</button>
                  </div>
              </div>

              <div className="sidebar">
                  <div className="players-info">
                      <div className={`player-info ${isMyTurn ? 'current-player' : ''}`}>
                          <h3>{myPlayerInfo?.username} (You)</h3>
                          <p>Colour: {myColor === 1 ? 'Black' : 'White'}</p>
                          <p>Thinking time: {formatTime(myPlayerInfo?.remainingTime)}s</p>
                          <p>Second reading: {myPlayerInfo?.currentByoyomiTime}s ({myPlayerInfo?.byoyomiPeriods}times)
                          </p>
                      </div>
                      <div className="player-info">
                          <h3>{opponentPlayerInfo.username} (Opponent - {opponentPlayerInfo.color === 1 ? 'Black' : 'White'})</h3>
                          <p>Colour: {opponentPlayerInfo?.color === 1 ? 'Black' : 'White'}</p>
                          <p>Thinking time: {formatTime(opponentPlayerInfo?.remainingTime)}s</p>
                          <p>{opponentPlayerInfo?.byoyomiPeriods > 0 &&
                              `Second reading: ${opponentPlayerInfo?.currentByoyomiTime}s (${opponentPlayerInfo?.byoyomiPeriods}times)`}
                          </p>
                      </div>
                  </div>
                  <div className="small-board-area">
                      <h4>Sub board</h4>
                      <GoBoardComponent
                          boardState={gameState.board}
                          boardSize={gameState.boardSize}
                          isSmall={true}
                          torusType={gameState.torusType}
                          onPlaceStone={() => {}}
                          isMyTurn={false}
                          offset={smallBoardOffset}
                          isClickable={false}
                      />
                      <div className="board-move-buttons small-board-buttons">
                          <button onClick={() => handleSmallBoardMove('col', 'single',false)}>←1</button>
                          <button onClick={() => handleSmallBoardMove('col', 'five',false)}>←5</button>
                          <button onClick={() => handleSmallBoardMove('col', 'single', true)}>→1</button>
                          <button onClick={() => handleSmallBoardMove('col', 'five', true)}>→5</button>
                          <button onClick={() => handleSmallBoardMove('row', 'single',false)} disabled={gameState.torusType === 'horizontal'}>↑1</button>
                          <button onClick={() => handleSmallBoardMove('row', 'five',false)} disabled={gameState.torusType === 'horizontal'}>↑5</button>
                          <button onClick={() => handleSmallBoardMove('row', 'single', true)} disabled={gameState.torusType === 'horizontal'}>↓1</button>
                          <button onClick={() => handleSmallBoardMove('row', 'five', true)} disabled={gameState.torusType === 'horizontal'}>↓5</button>
                      </div>
                  </div>


                  {isGameEnded && (
                      <div className="post-game-actions">
                          <button onClick={handleReturnToLobby}>Return to Lobby</button>
                          <button onClick={handleDownloadKifu} disabled={!kifuData}>Download kifu</button>
                      </div>
                  )}
              </div>
          </div>
          {showEndGameConfirm && (
              <EndGameModal
                  message="The result will be judged only by the number of remaining stones.<br>Sure to request？"
                  onConfirm={confirmEndGameRequest}
                  onCancel={() => setShowEndGameConfirm(false)}
                  confirmText="Request"
                  cancelText="Cancel"
              />
          )}

          {showIncomingEndGameRequest && (
              <EndGameModal
                  message={`End Game Request from ${opponentPlayerInfo.username}<br>The result will be judged only by the number of remaining stones.<br>Sure to confirm？`}
                  onConfirm={() => handleRespondToEndGameRequest(true)}
                  onCancel={() => handleRespondToEndGameRequest(false)}
                  confirmText="Confirm"
                  cancelText="Decline"
              />
          )}
        </div>
    );
}

export default GameScreen;
