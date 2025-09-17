import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import './LobbyScreen.css';
import MatchRequestModal from '../components/MatchRequestModal';
import { useNavigate } from 'react-router-dom';
import IncomingMatchRequestModal from '../components/IncomingMatchRequestModal';
import AiMatchRequestModal from '../components/AiMatchRequestModal';
import SelfMatchRequestModal from '../components/SelfMatchRequestModal';

// This is required for TypeScript/some linters to know about particlesJS
//declare const particlesJS: any;

function LobbyScreen({ currentUser, socket, onLogout }) {
    const [lobbyUsers, setLobbyUsers] = useState([]);
    const [sortKey, setSortKey] = useState('username');
    const [sortDirection, setSortDirection] = useState('asc');
    const [showMatchModal, setShowMatchModal] = useState(false);
    const [showSelfMatchModal, setShowSelfMatchModal] = useState(false)
    const [selectedOpponent, setSelectedOpponent] = useState(null);
    const [incomingMatchRequest, setIncomingMatchRequest] = useState(null);
    const [showAiMatchModal, setShowAiMatchModal] = useState(false);
    const navigate = useNavigate();

    const SERVER_URL = process.env.REACT_APP_SERVER_URL || 'http://localhost:5000';

    const particlesConfig = {
      "particles":{
        "number":{
          "value":38,
          "density":{
            "enable":true,
            "value_area":800
          }
        },
        "color":{
          "value":"#000000"
        },
        "shape":{
          "type":"polygon",
          "stroke":{
            "width":0,
          },
          "polygon":{
            "nb_sides":3
          },
          "image":{
            "width":190,
            "height":100
          }
        },
        "opacity":{
        "value":0.664994832269074,
        "random":false,
        "anim":{
          "enable":true,
          "speed":2.2722661797524872,
          "opacity_min":0.08115236356258881,
          "sync":false
        }
        },
        "size":{
          "value":3,
          "random":true,
          "anim":{
            "enable":false,
            "speed":40,
            "size_min":0.1,
            "sync":false
          }
        },
        "line_linked":{
          "enable":true,
          "distance":150,
          "color":"#000000",
          "opacity":0.6,
          "width":1
        },
        "move":{
          "enable":true,
          "speed":6,
          "direction":"none",
          "random":false,
          "straight":false,
          "out_mode":"out",
          "bounce":true,
          "attract":{
            "enable":false,
            "rotateX":600,
            "rotateY":961.4383117143238
          }
        }
      },
      "interactivity":{
        "detect_on":"canvas",
        "events":{
          "onhover":{
            "enable":false,
            "mode":"repulse"
          },
          "onclick":{
            "enable":false
          },
          "resize":true
        }
      },
      "retina_detect":true
    };

    useEffect(() => {
        // The check for `typeof particlesJS !== 'undefined'` is crucial.
        if (typeof particlesJS !== 'undefined') {
            particlesJS("particles-js", particlesConfig);
        }
    }, []);

    const fetchLobbyUsers = useCallback(async () => {
        try {
            const token = localStorage.getItem('token');
            const res = await axios.get(`${SERVER_URL}/api/users/lobby`, {
                headers: { Authorization: `Bearer ${token}` }
            });

            //自分と対戦する用のユーザーオブジェクト
            const soloUser = {
                id: 'solo-player-id',
                username: 'Play by yourself',
                score: { wins: '-', losses: '-' },
                rating: '-',
                isPlaying: false,
                isGuest: false
            };

            const aiUser = {
                id: 'ai-player-id',
                username: 'AI',
                score: { wins: '-', losses: '-' },
                rating: '-',
                isPlaying: false,
                isGuest: false
            };
            //soloUserをAIより上に配置
            setLobbyUsers([soloUser, aiUser, ...res.data.filter(u => u.id !== currentUser.id)]);
        } catch (err) {
            console.error('Failed to fetch lobby users:', err.response?.data?.msg || err.message);
            onLogout();
        }
    }, [onLogout, SERVER_URL, currentUser.id]);

    useEffect(() => {
        const handleSocketConnect = () => {
            if (currentUser && currentUser.id) {
                socket.emit('set_user_id', currentUser.id);
            }
        };

        const handleSocketDisconnect = () => {
            console.log(`[LobbyScreen] Socket DISCONNECTED.`);
        };

        socket.on('connect', handleSocketConnect);
        socket.on('disconnect', handleSocketDisconnect);

        if (currentUser && currentUser.id && socket.connected) {
            socket.emit('set_user_id', currentUser.id);
        }

        fetchLobbyUsers();
        socket.on('update_lobby_users', (users) => {
          //soloUserをAIより上に配置
          const soloUser = {
                id: 'solo-player-id',
                username: 'Play by yourself',
                score: { wins: '-', losses: '-' },
                rating: '-',
                isPlaying: false,
                isGuest: false
            };
          const aiUser = {
              id: 'ai-player-id',
              username: 'AI',
              score: { wins: '-', losses: '-' },
              rating: '-',
              isPlaying: false,
              isGuest: false
          };
          setLobbyUsers([soloUser, aiUser, ...users.filter(u => u.id !== currentUser.id)]);
        });

        socket.on('receive_match_request', (data) => {
            setIncomingMatchRequest({ ...data, isCorrection: data.isCorrection || false });
            alert(`Match Request from ${data.requesterUsername}!`);
        });

        socket.on('match_accepted', ({ gameId, isPlayer1Black, initialGameState }) => {});

        socket.on('match_rejected', ({ respondentUsername }) => {
            alert(`${respondentUsername} declined the request`);
        });

        socket.on('start_game', (gameId) => {
            setIncomingMatchRequest(null);
            navigate(`/game/${gameId}`);
        });

        socket.on('match_request_failed', (data) => {
            alert(`Failed to request: ${data.message}`);
        });

        return () => {
            socket.off('connect', handleSocketConnect);
            socket.off('disconnect', handleSocketDisconnect);
            socket.off('update_lobby_users');
            socket.off('receive_match_request');
            socket.off('start_game');
            socket.off('match_request_failed');
            socket.off('match_rejected');
        };
    }, [socket, fetchLobbyUsers, navigate, currentUser.id]);

    const sortedUsers = [...lobbyUsers].sort((a, b) => {
        //soloUserとaiUserをソート対象外にし、常に一番上に表示
        if (a.id === 'solo-player-id') return -1;
        if (b.id === 'solo-player-id') return 1;
        if (a.id === 'ai-player-id') return -1;
        if (b.id === 'ai-player-id') return 1;
        let valA = a[sortKey];
        let valB = b[sortKey];
        if (sortKey === 'score') {
            valA = a.score.wins - a.score.losses;
            valB = b.score.wins - b.score.losses;
        }
        if (typeof valA === 'string') {
            return sortDirection === 'asc' ? valA.localeCompare(valB) : valB.localeCompare(valA);
        } else {
            return sortDirection === 'asc' ? valA - valB : valB - valA;
        }
    });

    const handleUserClick = (user) => {
        //soloUserがクリックされた場合の処理
        if (user.id === 'solo-player-id') {
            setShowSelfMatchModal(true);
            return;
        }

        if (user.id === currentUser.id) {
          return;
        }
        if (user.id === 'ai-player-id') {
            setShowAiMatchModal(true)
            return;
        }
        if (user.isPlaying) {
            alert(`${user.username} is now playing`);
            return;
        }
        if (!user.socketId) {
            alert(`${user.username} is offline`);
            return;
        }
        setSelectedOpponent(user);
        setShowMatchModal(true);
    };

    const handleSendMatchRequest = (settingsFromModal) => {
        if (!selectedOpponent) {
          return;
        }
        const finalSettings = {
            boardSize: settingsFromModal?.boardSize || 9,
            timeLimit: settingsFromModal?.timeLimit || 10,
            byoyomiTime: settingsFromModal?.byoyomiTime || 30,
            byoyomiCount: settingsFromModal?.byoyomiCount || 3,
            torusType: settingsFromModal?.torusType || 'none',
        };
        socket.emit('match_request', {
            requesterId: currentUser.id,
            opponentId: selectedOpponent.id,
            ...finalSettings
        });
        setShowMatchModal(false);
        setSelectedOpponent(null);
        alert(`Send a match request to ${selectedOpponent.username}`);
    };

    const handleRespondToMatchRequest = useCallback((accepted, conditions) => {
        if (!incomingMatchRequest) return;
        socket.emit('match_response', {
            matchRequestId: incomingMatchRequest.matchRequestId,
            requestId: incomingMatchRequest.requesterId,
            respondentId: currentUser.id,
            isAccepted: accepted,
            initialSettings: conditions.initialSettings,
            modifiedSettings: conditions.modifiedSettings
        });
        setIncomingMatchRequest(null);
    }, [socket, incomingMatchRequest, currentUser.id]);

    const handleCorrectMatchRequest = useCallback((correctionDetails) => {
        if (incomingMatchRequest) {
            socket.emit('match_response', {
                matchRequestId: incomingMatchRequest.matchRequestId,
                requestId: incomingMatchRequest.requesterId,
                respondentId: currentUser.id,
                isAccepted: false,
                isCorrection: true,
                initialSettings: correctionDetails.initialSettings,
                modifiedSettings: correctionDetails.modifiedSettings
            });
            setIncomingMatchRequest(null);
            alert(`Send modified request to ${incomingMatchRequest.requesterUsername}`);
        }
    }, [socket, incomingMatchRequest, currentUser.id]);

    const handleStartAiMatch = useCallback((conditions) => {
        socket.emit('start_ai_game', {
            playerId: currentUser.id,
            boardSize: conditions.boardSize,
            timeLimit: conditions.timeLimit,
            byoyomiTime: conditions.byoyomiTime,
            byoyomiCount: conditions.byoyomiCount,
            torusType: conditions.torusType,
        });
        setShowAiMatchModal(false);
    }, [socket, showAiMatchModal, currentUser.id]);

    const handleStartSelfMatch = useCallback((conditions) => {
        socket.emit('start_self_match', {
            playerId: currentUser.id,
            boardSize: conditions.boardSize,
            torusType: conditions.torusType,
        });
        setShowSelfMatchModal(false);
    }, [socket, showSelfMatchModal, currentUser.id]);


    const filteredLobbyUsers = sortedUsers.filter(user =>
        user.id !== currentUser.id && (user.socketId || user.id === 'ai-player-id' || user.id === 'solo-player-id')
    );

    return (
        <div className="lobby-screen-container">
            {/* White background layer */}
            <div className="particles-background"></div>
            {/* Particle canvas layer */}
            <div id="particles-js"></div>

            <header className="lobby-header">
                <h1>Torus Go Online - {currentUser.username}</h1>
                <button onClick={onLogout} className="logout-button">Sign out</button>
            </header>
            <div className="user-list-section">
                <h2>List of online users</h2>
                <div className="user-list-table-container">
                    <table className="user-list-table">
                        <thead>
                            <tr>
                                <th onClick={() => handleSort('username')}>User name {sortKey === 'username' && (sortDirection === 'asc' ? '▲' : '▼')}</th>
                                <th onClick={() => handleSort('score')}>score {sortKey === 'score' && (sortDirection === 'asc' ? '▲' : '▼')}</th>
                                <th onClick={() => handleSort('rating')}>Rating {sortKey === 'rating' && (sortDirection === 'asc' ? '▲' : '▼')}</th>
                                <th onClick={() => handleSort('isPlaying')}>Status {sortKey === 'isPlaying' && (sortDirection === 'asc' ? '▲' : '▼')}</th>
                            </tr>
                        </thead>
                        <tbody>
                            {sortedUsers.map((user) => (
                                <tr
                                    key={user.id}
                                    className={`user-row ${user.isPlaying ? 'playing' : ''} ${user.id === currentUser.id ? 'current-user' : ''}`}
                                    onClick={() => handleUserClick(user)}
                                >
                                    <td>{user.username} {user.isGuest && '(guest)'} {user.id === currentUser.id && '(あなた)'}</td>
                                    <td>{typeof user.score.wins === 'number' ? `${user.score.wins}-${user.score.losses}` : user.score.wins}</td>
                                    <td>{user.rating}</td>
                                    <td>{user.isPlaying ? 'Playing' : 'Waiting'}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
            {showMatchModal && selectedOpponent &&(
                <MatchRequestModal
                    opponent={selectedOpponent}
                    onClose={() => setShowMatchModal(false)}
                    onSendRequest={handleSendMatchRequest}
                />
            )}
            {incomingMatchRequest && (
              <IncomingMatchRequestModal
                  requestDetails={incomingMatchRequest}
                  onConfirm={(conditions) => handleRespondToMatchRequest(true, conditions)}
                  onCancel={() => handleRespondToMatchRequest(false, {})}
                  onCorrect={handleCorrectMatchRequest}
              />
            )}
            {showAiMatchModal && (
              <AiMatchRequestModal
                  onClose={() => setShowAiMatchModal(false)}
                  onConfirm={(conditions) => handleStartAiMatch(conditions)}
              />
            )}
            {showSelfMatchModal && (
              <SelfMatchRequestModal
                  onClose={() => setShowSelfMatchModal(false)}
                  onConfirm={(conditions) => handleStartSelfMatch(conditions)}
              />
            )}
        </div>
    );
}

export default LobbyScreen;
