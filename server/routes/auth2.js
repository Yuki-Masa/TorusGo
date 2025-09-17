// server/routes/auth.js

const express = require('express');
const router = express.Router();
const jwt = require('jsonwebtoken');
const User = require('../models/User');

const generateToken = (id) => {
    return jwt.sign({ id }, process.env.JWT_SECRET, {
        expiresIn: '1h'
    });
};

// Register
router.post('/register', async (req, res) => {
    const { username, password } = req.body;
    try {
        let user = await User.findOne({ username });
        if (user) {
            return res.status(400).json({ msg: 'ユーザー名は既に使用されています' });
        }
        user = new User({ username, password });
        await user.save();
        res.status(201).json({
            msg: `登録完了 ${user.username}さん、ようこそ！`,
            user: { id: user._id, username: user.username }
        });
    } catch (err) {
        console.error(err.message);
        res.status(500).send('サーバーエラーが発生しました');
    }
});

// Login
router.post('/login', async (req, res) => {
    const { username, password } = req.body;
    try {
        const user = await User.findOne({ username }).select('+password');
        if (!user) {
            return res.status(400).json({ msg: '無効な認証情報です' });
        }
        const isMatch = await user.matchPassword(password);
        if (!isMatch) {
            return res.status(400).json({ msg: '無効な認証情報です' });
        }
        res.json({
            token: generateToken(user._id),
            user: {
                id: user._id,
                username: user.username,
                score: user.score,
                rating: user.rating,
                is_playing: user.is_playing,
                is_guest: user.is_guest
            }
        });
    } catch (err) {
        console.error(err.message);
        res.status(500).send('サーバーエラーが発生しました');
    }
});

// Guest Play
router.post('/guest-play', async (req, res) => {
    try {
        let guestUsername;
        let userExists = true;
        while (userExists) {
            const randomNumber = Math.floor(1000 + Math.random() * 9000);
            guestUsername = `guest${randomNumber}`;
            const existingUser = await User.findOne({ username: guestUsername });
            if (!existingUser) {
                userExists = false;
            }
        }
        const guestUser = new User({
            username: guestUsername,
            password: Math.random().toString(36).substring(2, 15), // ダミーパスワード
            is_guest: true
        });
        await guestUser.save();
        res.json({
            token: generateToken(guestUser._id),
            user: {
                id: guestUser._id,
                username: guestUser.username,
                score: guestUser.score,
                rating: guestUser.rating,
                is_playing: guestUser.is_playing,
                is_guest: guestUser.is_guest
            }
        });
    } catch (err) {
        console.error(err.message);
        res.status(500).send('サーバーエラーが発生しました');
    }
});

module.exports = router;
