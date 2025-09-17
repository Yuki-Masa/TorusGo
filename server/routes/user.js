// server/routes/user.js

const express = require('express');
const router = express.Router();
const User = require('../models/User');
const protect = require('../middleware/auth');

router.get('/lobby', protect, async (req, res) => {
    try {
        const users = await User.find({})
            .select('username score rating is_playing is_guest socket_id');
        const lobbyUsers = users.map(user => ({
            id: user._id,
            username: user.username,
            score: user.score,
            rating: user.rating,
            isPlaying: user.is_playing,
            isGuest: user.is_guest,
            socketId: user.socket_id // ★修正: socket_id を追加
        }));
        res.json(lobbyUsers);
    } catch (err) {
        console.error(err.message);
        res.status(500).send('サーバーエラーが発生しました');
    }
});

module.exports = router;
