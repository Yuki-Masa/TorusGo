// client/src/pages/InitialScreen.js

import React, { useState } from 'react';
import axios from 'axios';
import './InitialScreen.css';

function InitialScreen({ onLogin }) {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [message, setMessage] = useState('');
    const [isRegisterMode, setIsRegisterMode] = useState(false);

    const SERVER_URL = process.env.REACT_APP_SERVER_URL || 'http://localhost:5000';

    const handleSubmit = async (e) => {
        e.preventDefault();
        setMessage('');

        if (!username || !password) {
            setMessage('Please enter Username and Password');
            return;
        }

        try {
            if (isRegisterMode) {
                const res = await axios.post(`${SERVER_URL}/api/auth/register`, { username, password });
                setMessage(res.data.msg);
                setIsRegisterMode(false); // 登録後ログイン画面に戻る
                setUsername('');
                setPassword('');
            } else {
                const res = await axios.post(`${SERVER_URL}/api/auth/login`, { username, password });
                onLogin(res.data.user, res.data.token);
            }
        } catch (err) {
            console.error(err.response ? err.response.data : err);
            setMessage(err.response?.data?.msg || 'Failed to authenticate');
        }
    };

    const handleGuestPlay = async () => {
        setMessage('');
        try {
            const res = await axios.post(`${SERVER_URL}/api/auth/guest-play`);
            onLogin(res.data.user, res.data.token);
        } catch (err) {
            console.error(err.response ? err.response.data : err);
            setMessage(err.response?.data?.msg || 'Failed to start guest mode');
        }
    };

    return (
        <div className="initial-screen-container">

            <div className="initial-screen-box">
                <h2>{isRegisterMode ? 'Sign up' : 'Sign in'}</h2>
                <form onSubmit={handleSubmit}>
                    <div className="form-group">
                        <label htmlFor="username">User name:</label>
                        <input
                            type="text"
                            id="username"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label htmlFor="password">Password:</label>
                        <input
                            type="password"
                            id="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                        />
                    </div>
                    {message && <p className="message">{message}</p>}
                    <button type="submit" className="primary-button">
                        {isRegisterMode ? 'Sign up' : 'Sign in'}
                    </button>
                </form>
                <div className="toggle-mode">
                    <button onClick={() => setIsRegisterMode(!isRegisterMode)} className="secondary-button">
                        {isRegisterMode ? 'Already have an account?' : 'Haven’t created an account?'}
                    </button>
                    <button onClick={handleGuestPlay} className="guest-button">
                        Play as a guest
                    </button>
                </div>
            </div>
        </div>
    );
}

export default InitialScreen;
