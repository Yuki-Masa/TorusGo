// server/websocket/handlers.js

const { GameManager } = require('../game/GameLogic');
const User = require('../models/User');
const axios = require('axios'); // AI連携用

const connectedUsers = new Map(); // <userId, socketId>
const games = new Map(); // <gameId, GameManagerInstance>
//進行中の対局交渉を管理するマップ
const ongoingMatchNegotiations = new Map(); // <matchRequestId, { requesterId, opponentId, latestSettings }>

// ヘルパー関数: ロビーユーザーリストを全クライアントにブロードキャスト
async function broadcastLobbyUsers(io) {
    try {
        const users = await User.find({})
            .select('username score rating is_playing is_guest socket_id');
        const lobbyUsersData = users.map(user => ({
            id: user._id,
            username: user.username,
            score: user.score,
            rating: user.rating,
            isPlaying: user.is_playing,
            isGuest: user.is_guest,
            socketId: user.socket_id
        }));
        io.emit('update_lobby_users', lobbyUsersData);
    } catch (err) {
        console.error('Error broadcasting lobby users:', err);
    }
}

module.exports = (io) => {
    io.on('connection', async (socket) => {
        console.log(`User connected: ${socket.id}`);

        socket.on('player_leaving_game', async (data) => {
            const { gameId, userId } = data;
            console.log(`[Server] Player ${userId} is leaving game ${gameId}.`);
            try {
                // ユーザーのis_playingステータスを更新
                await User.findByIdAndUpdate(userId, { is_playing: false });
                console.log(`[Server] User ${userId} is_playing status set to false.`);

                // ゲームの状態を確認し、必要に応じてクリーンアップ
                const gameManager = games.get(gameId);
                if (gameManager) {

                    // 現状はシンプルに、ゲームから離れたプレイヤーのis_playingをfalseに
                    // もし両プレイヤーが離れた場合、ゲームを削除するなどの処理も検討
                    const player1Id = gameManager.player1.id;
                    const player2Id = gameManager.player2.id;

                    const player1User = await User.findById(player1Id);
                    const player2User = await User.findById(player2Id);

                    let otherPlayerStillPlaying = false;
                    if (player1User && player1User.is_playing && player1Id !== userId) {
                        otherPlayerStillPlaying = true;
                    }
                    if (player2User && player2User.is_playing && player2Id !== userId) {
                        otherPlayerStillPlaying = true;
                    }

                    if (!otherPlayerStillPlaying && gameManager.player2.id !== 'ai-player-id') {
                        // 両方の人間プレイヤーがゲームを離れた場合、ゲームを終了または削除
                        console.log(`[Server] Both human players left game ${gameId}. Deleting game.`);
                        //ゲームが既に終了している場合はendGameを呼ばない
                        if (!gameManager.gameEnded) {
                            gameManager.endGame('players_left', userId, io); // ゲーム終了イベントを発火
                        }
                        games.delete(gameId); // ゲームマネージャーを削除
                    } else if (gameManager.player2.id === 'ai-player-id' && userId === player1Id) {
                        // AI対局で人間プレイヤーが離れた場合
                        console.log(`[Server] Human player left AI game ${gameId}. Deleting game.`);
                        // ゲームが既に終了している場合はendGameを呼ばない
                        if (!gameManager.gameEnded) {
                             gameManager.endGame('player_left_ai_game', userId, io); // AI対局終了イベント
                        }
                        games.delete(gameId); // ゲームマネージャーを削除
                    }
                } else {
                    console.log(`[Server] Game ${gameId} not found when player left.`);
                }

                await broadcastLobbyUsers(io); // ロビーユーザーリストを更新して、is_playingの変化を反映
            } catch (error) {
                console.error(`[Server] Error handling player_leaving_game for user ${userId} in game ${gameId}:`, error);
            }
        });

        socket.on('set_user_id', async (userId) => {
            console.log(`[set_user_id] Received event for userId: ${userId}, socket.id: ${socket.id}`);
            if (userId) {
                const oldSocketId = connectedUsers.get(userId);
                if (oldSocketId && oldSocketId !== socket.id) {
                    console.log(`[set_user_id] userId ${userId} was previously associated with socket ${oldSocketId}. Updating to ${socket.id}`);
                } else if (!oldSocketId) {
                    console.log(`[set_user_id] userId ${userId} not found in map. Adding new entry for socket ${socket.id}`);
                }
                connectedUsers.set(userId, socket.id);
                console.log(`[set_user_id] connectedUsers map updated. Current map size: ${connectedUsers.size}`);

                try {
                    const user = await User.findById(userId);
                    if (user) {
                        if (user.socket_id !== socket.id) {
                            console.log(`[set_user_id] DB: Updating socket_id for user ${user.username} from ${user.socket_id} to ${socket.id}`);
                            user.socket_id = socket.id;
                            await user.save();
                            console.log(`[set_user_id] DB: Successfully updated socket_id for user ${user.username}.`);
                        } else {
                            console.log(`[set_user_id] DB: socket_id for user ${user.username} is already ${socket.id}. No update needed.`);
                        }
                        socket.join(userId.toString());
                        console.log(`[set_user_id] User ${user.username} joined room ${userId.toString()}.`);
                    } else {
                        console.log(`[set_user_id] WARN: User with ID ${userId} not found in DB.`);
                    }
                } catch (err) {
                    console.error(`[set_user_id] ERROR: Failed to set user ID or update socket ID for ${userId}:`, err);
                }
            } else {
                console.log(`[set_user_id] WARN: Received set_user_id event with null/undefined userId.`);
            }
            setTimeout(async () => {
                await broadcastLobbyUsers(io);
                console.log(`[set_user_id] Broadcasted lobby users after set_user_id.`);
            }, 100);
        });

        // 対局申し込み
        socket.on('match_request', async (data) => {
            const { requesterId, opponentId, boardSize, timeLimit, byoyomiTime, byoyomiCount, torusType } = data;
            const opponentSocketId = connectedUsers.get(opponentId);

            try {
                const requesterUser = await User.findById(requesterId).select('username is_playing');
                const opponentUser = await User.findById(opponentId).select('username is_playing');

                if (!requesterUser) {
                    io.to(socket.id).emit('match_request_failed', { message: 'Cannot find the requester' });
                    return;
                }
                if (!opponentUser) {
                    io.to(socket.id).emit('match_request_failed', { message: 'Cannot find the user' });
                    return;
                }

                if (requesterUser.is_playing || opponentUser.is_playing) {
                     io.to(socket.id).emit('match_request_failed', { message: 'User is in a game' });
                     return;
                }

                if (opponentSocketId) {
                    console.log('Received match_request data:', data);

                    // ユニークなmatchRequestIdを生成し、交渉状態を保存
                    const matchRequestId = `negotiation_${Date.now()}_${Math.random().toString(36).substring(2, 8)}`;
                    const initialSettings = { boardSize, timeLimit, byoyomiTime, byoyomiCount, torusType };
                    ongoingMatchNegotiations.set(matchRequestId, {
                        requesterId,
                        opponentId,
                        latestSettings: initialSettings // 初期設定を最新の設定として保存
                    });
                    console.log(`[Server] New negotiation started. matchRequestId: ${matchRequestId}`);
                    console.log(`[Server] ongoingMatchNegotiations size: ${ongoingMatchNegotiations.size}`);


                    io.to(opponentSocketId).emit('receive_match_request', {
                        matchRequestId, // クライアントに交渉IDを渡す
                        requesterId: requesterId,
                        requesterUsername: requesterUser.username,
                        boardSize,
                        timeLimit,
                        byoyomiTime,
                        byoyomiCount,
                        torusType
                    });
                    console.log(`Match request from ${requesterUser.username} to ${opponentUser.username}`);
                } else {
                    io.to(socket.id).emit('match_request_failed', { message: 'The user is offline' });
                }
            } catch (err) {
                console.error('Error handling match_request:', err);
                io.to(socket.id).emit('match_request_failed', { message: 'Error' });
            }
        });

        // 対局申し込みへの応答
        socket.on('match_response', async (data) => {
            //matchRequestId を受け取る
            const { matchRequestId, requestId, respondentId, isAccepted, modifiedSettings, initialSettings, isCorrection } = data;
            const requesterSocketId = connectedUsers.get(requestId);
            const respondentSocketId = connectedUsers.get(respondentId); // 応答者のソケットIDも取得

            console.log('Received match_response data:', data);
            console.log(`[Server] matchRequestId: ${matchRequestId}`);

            // ongoingMatchNegotiationsから交渉状態を取得
            const negotiation = ongoingMatchNegotiations.get(matchRequestId);

            if (!negotiation) {
                console.error(`[Server] Negotiation with matchRequestId ${matchRequestId} not found.`);
                io.to(socket.id).emit('match_request_failed', { message: 'Cannot find the user' });
                return;
            }

            try {
                const requesterUser = await User.findById(requestId).select('username is_playing');
                const respondentUser = await User.findById(respondentId).select('username is_playing');

                if (!requesterUser) {
                    console.error(`User (requester) not found for ID: ${requestId}`);
                    io.to(socket.id).emit('match_request_failed', { message: 'Cannot find the requester' });
                    return;
                }
                if (!respondentUser) {
                    console.error(`User (respondent) not found for ID: ${respondentId}`);
                    io.to(socket.id).emit('match_request_failed', { message: 'Cannot find the user' });
                    return;
                }

                if (requesterUser.is_playing || respondentUser.is_playing) {
                    console.log('One of the players is already playing. Denying match.');
                    if (requesterSocketId) {
                        io.to(requesterSocketId).emit('match_request_failed', { message: 'The user is in a game' });
                    }
                    io.to(socket.id).emit('match_request_failed', { message: 'You are already in a game' });
                    // negotiation をクリーンアップ
                    ongoingMatchNegotiations.delete(matchRequestId);
                    console.log(`[Server] ongoingMatchNegotiations size after played cleanup: ${ongoingMatchNegotiations.size}`);
                    return;
                }

                if (isCorrection) {
                    // 修正申し込みの場合
                    console.log(`Match correction request from ${respondentUser.username} to ${requesterUser.username}`);

                    // negotiationのlatestSettingsを更新
                    negotiation.latestSettings = modifiedSettings;
                    ongoingMatchNegotiations.set(matchRequestId, negotiation); // マップを更新

                    if (requesterSocketId) {
                        io.to(requesterSocketId).emit('receive_match_request', {
                            matchRequestId, // クライアントに交渉IDを渡す
                            requesterId: respondentId, // 修正申し込みを送信したのが respondent なので、requesterIdは respondentId
                            requesterUsername: respondentUser.username,
                            boardSize: modifiedSettings.boardSize,
                            timeLimit: modifiedSettings.timeLimit,
                            byoyomiTime: modifiedSettings.byoyomiTime,
                            byoyomiCount: modifiedSettings.byoyomiCount,
                            torusType: modifiedSettings.torusType,
                            isCorrection: true // 修正申し込みであることを示す
                        });
                        console.log(`[Server] Emitted receive_match_request (correction) to ${requesterUser.username} with latest settings.`);
                    }
                    return; // 修正申し込みの場合はゲーム開始しない
                }

                if (isAccepted) {
                    // 通常の承諾の場合
                    // GameManagerに渡す設定は ongoingMatchNegotiations から取得
                    const finalSettings = negotiation.latestSettings;

                    const gameId = `game_${Date.now()}_${Math.random().toString(36).substring(2, 8)}`;

                    console.log('[Server] Final settings for GameManager (from negotiation):', finalSettings);
                    console.log('[Server] Board size passed to GameManager:', finalSettings.boardSize);

                    await User.findByIdAndUpdate(requestId, { is_playing: true });
                    await User.findByIdAndUpdate(respondentId, { is_playing: true });

                    const gameManager = new GameManager(
                        gameId,
                        requestId,
                        respondentId,
                        requesterUser.username,
                        respondentUser.username,
                        finalSettings.boardSize,
                        finalSettings.torusType || 'none',
                        finalSettings.timeLimit,
                        finalSettings.byoyomiTime,
                        finalSettings.byoyomiCount
                    );
                    games.set(gameId, gameManager);
                    gameManager.startGame(io);

                    // socket.ioのroom機能を使って、両方のプレイヤーをゲームのroomに参加させる
                    if (requesterSocketId) {
                        io.to(requesterSocketId).socketsJoin(gameId);
                        io.to(requesterSocketId).emit('start_game', gameId);
                    }
                    if (respondentSocketId) {
                        io.to(respondentSocketId).socketsJoin(gameId);
                        io.to(respondentSocketId).emit('start_game', gameId);
                    }

                    // negotiation をクリーンアップ
                    ongoingMatchNegotiations.delete(matchRequestId);
                    console.log(`[Server] ongoingMatchNegotiations size after game start: ${ongoingMatchNegotiations.size}`);

                    await broadcastLobbyUsers(io);
                    console.log(`Game ${gameId} started between ${requesterUser.username} and ${respondentUser.username}`);

                } else {
                    // 拒否の場合
                    if (requesterSocketId) {
                        io.to(requesterSocketId).emit('match_rejected', {
                            respondentUsername: respondentUser.username
                        });
                    }
                    console.log(`Match request from ${requesterUser.username} to ${respondentUser.username} rejected.`);
                    // negotiation をクリーンアップ
                    ongoingMatchNegotiations.delete(matchRequestId);
                    console.log(`[Server] ongoingMatchNegotiations size after rejection: ${ongoingMatchNegotiations.size}`);
                }
            } catch (err) {
                console.error('Error handling match_response:', err);
                io.to(socket.id).emit('match_request_failed', { message: 'An error occured while responding' });
            }
        });

        socket.on('start_self_match', async (data) => {
            const { playerId, boardSize, torusType} = data;
            try {

                const gameId = `self_game_${Date.now()}_${Math.random().toString(36).substring(2, 8)}`;
                const gameManager = new GameManager(
                    gameId,
                    playerId,
                    playerId,
                    player.username,
                    player.username,
                    boardSize,
                    torusType,
                );
                games.set(gameId, gameManager);
                gameManager.startGame(io);

                await User.findByIdAndUpdate(playerId, { is_playing: true });

                socket.join(gameId);
                io.to(socket.id).emit('start_game', gameId);
                await broadcastLobbyUsers(io);
                console.log(`Self Game ${gameId} started with ${player.username}`);

            } catch (err) {
                console.error('Error starting Self game:', err);
                io.to(socket.id).emit('game_start_failed', { message: 'An error occured while starting the game' });
            }
        });


        // AIとの対局開始
        socket.on('start_ai_game', async (data) => {
            const { playerId, boardSize, torusType , timeLimit, byoyomiTime, byoyomiCount } = data;
            try {
                const player = await User.findById(playerId).select('username is_playing');
                if (!player) {
                    io.to(socket.id).emit('game_start_failed', { message: 'Cannot find the user' });
                    return;
                }
                if (player.is_playing) {
                     io.to(socket.id).emit('game_start_failed', { message: 'You are already in a game' });
                     return;
                }

                const gameId = `ai_game_${Date.now()}_${Math.random().toString(36).substring(2, 8)}`;
                const gameManager = new GameManager(
                    gameId,
                    playerId,
                    'ai-player-id',
                    player.username,
                    'AI',
                    boardSize,
                    torusType,
                    timeLimit,
                    byoyomiTime,
                    byoyomiCount
                );
                games.set(gameId, gameManager);
                gameManager.startGame(io);

                await User.findByIdAndUpdate(playerId, { is_playing: true });

                socket.join(gameId);
                io.to(socket.id).emit('start_game', gameId);
                await broadcastLobbyUsers(io);
                console.log(`AI Game ${gameId} started with ${player.username}`);

            } catch (err) {
                console.error('Error starting AI game:', err);
                io.to(socket.id).emit('game_start_failed', { message: 'An error occured while starting the game' });
            }
        });

        // 以下、make_move, pass_turn, resign, request_end_game, respond_to_end_game, disconnect は変更なし
        // ゲーム状態のリクエスト
        socket.on('request_game_state', (gameId) => {
            const gameManager = games.get(gameId);
            if (gameManager) {
                io.to(socket.id).emit('update_game_state', gameManager.getGameState());
            } else {
                console.warn(`Game ${gameId} not found for state request.`);
                // ゲームが本当に見つからない場合のみgame_not_foundを発行
                io.to(socket.id).emit('game_not_found');
            }
        });

        // 着手
        socket.on('make_move', async (data) => {
            const { gameId, userId, row, col } = data;
            const gameManager = games.get(gameId);
            if (!gameManager) return;

            const currentPlayer = gameManager.currentTurn === gameManager.player1.color ? gameManager.player1 : gameManager.player2;

            const moveResult = gameManager.makeMove(row, col, userId, io);

            if (moveResult.success) {
                // AIのターンであればAIに処理を依頼
                if (gameManager.player2.id === 'ai-player-id' && gameManager.currentTurn === gameManager.player2.color) {
                    console.log('AI turn, requesting AI move...');
                    // AI_URLが設定されているかチェック
                    if (!process.env.AI_URL) {
                        console.error('Error: AI_URL environment variable is not set.');
                        gameManager.endGame('resignation', gameManager.player1.id, io); // AI_URLがない場合はゲーム終了
                        return;
                    }
                    try {
                        const aiResponse = await axios.post(process.env.AI_URL, {
                            gameId: gameId,
                            boardState: gameManager.board.board,
                            boardSize: gameManager.board.size,
                            torusType: gameManager.board.torusType,
                            currentTurn: gameManager.currentTurn, // AIのターンは GameManager.currentTurn を使う
                            koPoint: gameManager.board.koPoint
                        });

                        const aiMove = aiResponse.data.move;
                        if (aiMove === 'pass') {
                            gameManager.passTurn(gameManager.player2.id, io);
                        } else if (Array.isArray(aiMove) && aiMove.length === 2) {
                            const [aiRow, aiCol] = aiMove;
                            gameManager.makeMove(aiRow, aiCol, gameManager.player2.id, io);
                        } else {
                            console.error('Invalid AI move format:', aiMove);
                            gameManager.endGame('resignation', gameManager.player1.id, io); // 不正なAI動作でゲーム終了
                        }
                    } catch (aiError) {
                        // aiError.responseが存在しない場合はaiErrorオブジェクト全体をログに出力
                        console.error('Error requesting AI move:', aiError.response ? JSON.stringify(aiError.response.data) : aiError);
                        gameManager.endGame('resignation', gameManager.player1.id, io); // エラー発生でゲーム終了
                    }
                }
                io.to(gameId).emit('update_game_state', gameManager.getGameState());
            } else {
                io.to(socket.id).emit('move_failed', { message: moveResult.error });
            }
        });

        // パス
        socket.on('pass_turn', async (data) => {
            const { gameId, userId } = data;
            const gameManager = games.get(gameId);
            if (!gameManager) return;

            const passResult = gameManager.passTurn(userId, io);
            if (passResult.success) {
                if (passResult.gameEnded) {
                    // ゲーム終了処理はGameManager.endGameで発行される
                } else {
                    if (gameManager.player2.id === 'ai-player-id' && gameManager.currentTurn === gameManager.player2.color) {
                        console.log('AI turn after pass, requesting AI move...');
                        //AI_URLが設定されているかチェック
                        if (!process.env.AI_URL) {
                            console.error('Error: AI_URL environment variable is not set.');
                            gameManager.endGame('resignation', gameManager.player1.id, io); // AI_URLがない場合はゲーム終了
                            return;
                        }
                        try {
                            const aiResponse = await axios.post(process.env.AI_URL, {
                                gameId: gameId,
                                boardState: gameManager.board.board,
                                boardSize: gameManager.board.size,
                                torusType: gameManager.board.torusType,
                                currentTurn: gameManager.currentTurn, // AIのターンは GameManager.currentTurn を使うべき
                                koPoint: gameManager.board.koPoint
                            });

                            const aiMove = aiResponse.data.move;
                            if (aiMove === 'pass') {
                                gameManager.passTurn(gameManager.player2.id, io);
                            } else if (Array.isArray(aiMove) && aiMove.length === 2) {
                                const [aiRow, aiCol] = aiMove;
                                gameManager.makeMove(aiRow, aiCol, gameManager.player2.id, io);
                            } else {
                                console.error('Invalid AI move format:', aiMove);
                                gameManager.endGame('resignation', gameManager.player1.id, io); // 不正なAI動作でゲーム終了
                            }
                        } catch (aiError) {
                             // aiError.responseが存在しない場合はaiErrorオブジェクト全体をログに出力
                             console.error('Error requesting AI move after pass:', aiError.response ? JSON.stringify(aiError.response.data) : aiError);
                             gameManager.endGame('resignation', gameManager.player1.id, io); // エラー発生でゲーム終了
                        }
                    }
                    io.to(gameId).emit('update_game_state', gameManager.getGameState());
                }
            } else {
                io.to(socket.id).emit('action_failed', { message: passResult.error });
            }
        });

        // 投了
        socket.on('resign', (data) => {
            const { gameId, userId } = data;
            const gameManager = games.get(gameId);
            if (!gameManager) return;

            const resignResult = gameManager.resign(userId, io);
            if (!resignResult.success) {
                io.to(socket.id).emit('action_failed', { message: resignResult.error });
            }
        });

        // 終局申請
        socket.on('request_end_game', (data) => {
            const { gameId, userId } = data;
            const gameManager = games.get(gameId);
            if (!gameManager) return;

            const requestResult = gameManager.requestEndGame(userId, io);
            if (!requestResult.success) {
                io.to(socket.id).emit('action_failed', { message: requestResult.error });
            } else {
                io.to(gameId).emit('update_game_state', gameManager.getGameState());
            }
        });

        // 終局申請への応答
        socket.on('respond_to_end_game', (data) => {
            const { gameId, userId, agree } = data;
            const gameManager = games.get(gameId);
            if (!gameManager) return;

            const responseResult = gameManager.respondToEndGame(userId, agree, io);
            if (!responseResult.success) {
                io.to(socket.id).emit('action_failed', { message: responseResult.error });
            } else if (responseResult.resumed) {
                io.to(gameId).emit('update_game_state', gameManager.getGameState());
            }
        });


        socket.on('disconnect', async () => {
            console.log(`User disconnected: ${socket.id}`);
            let disconnectedUserId = null;
            for (const [userId, sockId] of connectedUsers.entries()) {
                if (sockId === socket.id) {
                    disconnectedUserId = userId;
                    connectedUsers.delete(userId);
                    break;
                }
            }

            if (disconnectedUserId) {
                try {
                    const user = await User.findById(disconnectedUserId);
                    if (user) {
                        // ゲストユーザーは削除、登録ユーザーはis_playingをfalseに
                        if (user.is_guest) {
                            await User.findByIdAndDelete(disconnectedUserId);
                            console.log(`Guest user ${user.username} deleted.`);
                        } else {
                            //切断時、is_playingはfalseにするが、socket_idはnullにする
                            await User.findByIdAndUpdate(disconnectedUserId, { socket_id: null, is_playing: false });
                        }
                    }

                    // 進行中のゲームがあれば、相手プレイヤーに切断を通知してゲームを終了
                    for (const [gameId, gameManager] of games.entries()) {
                        if (gameManager.player1.id === disconnectedUserId || gameManager.player2.id === disconnectedUserId) {
                            //ゲームが既に終了している場合は、切断による勝敗判定は行わない
                            if (!gameManager.gameEnded) {
                                // AI対局の場合はAIのターンでなければ、AI側のユーザーを勝者とする
                                if (gameManager.player2.id === 'ai-player-id' && gameManager.player1.id === disconnectedUserId) {
                                    gameManager.endGame('player_left_ai_game', disconnectedUserId, io); // 離脱したプレイヤーを渡す
                                } else if (gameManager.player1.id === disconnectedUserId && gameManager.player2.id !== 'ai-player-id') {
                                    gameManager.endGame('player_disconnected', gameManager.player2.id, io);
                                } else if (gameManager.player2.id === disconnectedUserId && gameManager.player1.id !== 'ai-player-id') {
                                    gameManager.endGame('player_disconnected', gameManager.player1.id, io);
                                }
                            }
                            //プレイヤーが離れたらゲームマネージャーを削除
                            games.delete(gameId);
                            console.log(`Game ${gameId} manager deleted due to player disconnect.`);
                            break; // 1つのゲームに関与していると仮定
                        }
                    }

                } catch (err) {
                    console.error('Error handling disconnect for user:', err);
                }
            }
            await broadcastLobbyUsers(io);
            // disconnect時に進行中の交渉をクリーンアップ (必要であれば)
            // このユーザーが関与している交渉を削除
            for (const [key, value] of ongoingMatchNegotiations.entries()) {
                if (value.requesterId === disconnectedUserId || value.opponentId === disconnectedUserId) {
                    ongoingMatchNegotiations.delete(key);
                    console.log(`[Server] Removed ongoing negotiation ${key} due to user disconnect.`);
                }
            }
        });
    });
};
