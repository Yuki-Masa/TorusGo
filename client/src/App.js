// client/src/App.js

import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import io from 'socket.io-client';
import InitialScreen from './pages/InitialScreen';
import LobbyScreen from './pages/LobbyScreen';
import GameScreen from './pages/GameScreen';
import './App.css';

// .env ファイルを読み込むためにREACT_APP_SERVER_URLを使用
const SERVER_URL = process.env.REACT_APP_SERVER_URL || 'http://localhost:5000';

const socket = io(SERVER_URL, {
    autoConnect: false // 自動接続しない
});

function AppContent() {
    const [currentUser, setCurrentUser] = useState(null);
    const navigate = useNavigate();

    useEffect(() => {
        const token = localStorage.getItem('token');
        const user = localStorage.getItem('user');

        if (token && user) {
            try {
                const parsedUser = JSON.parse(user);
                setCurrentUser(parsedUser);
                socket.auth = { token };
                socket.connect();

                socket.on('connect', () => {
                    console.log('Socket connected:', socket.id);
                    socket.emit('set_user_id', parsedUser.id);
                });

                // ロビー画面にリダイレクト
                navigate('/lobby');

            } catch (e) {
                console.error("Failed to parse user from localStorage", e);
                localStorage.clear();
                setCurrentUser(null);
                navigate('/');
            }
        } else {
            navigate('/');
        }

        // Socket.IOイベントリスナーをここで設定（コンポーネント全体で使うイベント）
        socket.on('connect_error', (err) => {
            console.error('Socket connection error:', err.message);
            // 認証エラーの場合、ログアウト処理
            if (err.message === 'Not authorized, token failed' || err.message === 'Not authorized, no token') {
                alert('セッションが切れました。再度ログインしてください。');
                handleLogout();
            }
        });

        return () => {
            if (socket.connected) {
                socket.disconnect();
            }
            socket.off('connect');
            socket.off('connect_error');
        };
    }, []); // 依存配列は空で、マウント時に一度だけ実行

    const handleLogin = (user, token) => {
        localStorage.setItem('token', token);
        localStorage.setItem('user', JSON.stringify(user));
        setCurrentUser(user);
        socket.auth = { token };
        if (!socket.connected) {
            socket.connect();
        } else {
            socket.emit('set_user_id', user.id); // 既に接続済みの場合はユーザーIDを再送
        }
        navigate('/lobby');
    };

    const handleLogout = () => {
        localStorage.clear();
        setCurrentUser(null);
        if (socket.connected) {
            socket.disconnect();
        }
        navigate('/');
    };

    return (
        <div className="App">
            <Routes>
                <Route path="/" element={<InitialScreen onLogin={handleLogin} />} />
                <Route path="/lobby" element={currentUser ? <LobbyScreen currentUser={currentUser} socket={socket} onLogout={handleLogout} /> : <InitialScreen onLogin={handleLogin} />} />
                <Route path="/game/:gameId" element={currentUser ? <GameScreen currentUser={currentUser} socket={socket} /> : <InitialScreen onLogin={handleLogin} />} />
                <Route path="/selfgame" element={currentUser ? <SelfScreen currentUser={currentUser} socket={socket} /> : <InitialScreen onLogin={handleLogin} />} />
                <Route path="*" element={<InitialScreen onLogin={handleLogin} />} />
            </Routes>
        </div>
    );
}

function App() {
    return (
        <Router>
            <AppContent />
        </Router>
    );
}

export default App;
