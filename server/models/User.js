// server/models/User.js

const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');

const UserSchema = new mongoose.Schema({
    username: {
        type: String,
        required: true,
        unique: true,
        trim: true
    },
    password: {
        type: String,
        required: true,
        select: false
    },
    score: {
        wins: { type: Number, default: 0 },
        losses: { type: Number, default: 0 }
    },
    rating: {
        type: Number,
        default: 1000
    },
    is_playing: {
        type: Boolean,
        default: false
    },
    is_guest: {
        type: Boolean,
        default: false
    },
    socket_id: {
        type: String,
        required: false // 接続中のソケットIDを保存
    }
}, { timestamps: true });

UserSchema.pre('save', async function(next) {
    if (!this.isModified('password')) return next();
    const salt = await bcrypt.genSalt(10);
    this.password = await bcrypt.hash(this.password, salt);
    next();
});

UserSchema.methods.matchPassword = async function(enteredPassword) {
    return await bcrypt.compare(enteredPassword, this.password);
};

module.exports = mongoose.model('User', UserSchema);
