// client/src/components/MatchRequestModal.js

import React, { useState } from 'react';
import './MatchRequestModal.css';

function MatchRequestModal({ opponent, onClose, onSendRequest }) {
    const [boardSize, setBoardSize] = useState(19);
    const [timeLimit, setTimeLimit] = useState(10);
    const [byoyomiTime, setByoyomiTime] = useState(30);
    const [byoyomiCount, setByoyomiCount] = useState(3);
    const [torusType, setTorusType] = useState('none'); // トーラスタイプを追加

    const handleSubmit = (e) => {
        e.preventDefault();
        onSendRequest({
            boardSize: parseInt(boardSize),
            timeLimit: parseInt(timeLimit),
            byoyomiTime: parseInt(byoyomiTime),
            byoyomiCount: parseInt(byoyomiCount),
            torusType: torusType // トーラスタイプも送信
        });
    };

    return (
        <div className="modal-overlay">
            <div className="modal-content">
                <h3>Match Request to {opponent.username}</h3>
                <form onSubmit={handleSubmit}>
                    <div className="form-group">
                        <label htmlFor="boardSize">Board size:</label>
                        <select id="boardSize" value={boardSize} onChange={(e) => setBoardSize(e.target.value)}>
                            <option value={6}>9×9</option>
                            <option value={9}>9×9</option>
                            <option value={13}>13×13</option>
                            <option value={17}>17×17</option>
                            <option value={19}>19×19</option>
                        </select>
                    </div>
                    <div className="form-group">
                        <label htmlFor="timeLimit">Thinking time:</label>
                        <input
                            type="number"
                            id="timeLimit"
                            value={timeLimit}
                            onChange={(e) => setTimeLimit(e.target.value)}
                            min="0"
                            max="60"
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label htmlFor="byoyomiTime">Second reading:</label>
                        <input
                            type="number"
                            id="byoyomiTime"
                            value={byoyomiTime}
                            onChange={(e) => setByoyomiTime(e.target.value)}
                            min="5"
                            max="60"
                            step="5"
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label htmlFor="byoyomiCount">Times of second reading:</label>
                        <input
                            type="number"
                            id="byoyomiCount"
                            value={byoyomiCount}
                            onChange={(e) => setByoyomiCount(e.target.value)}
                            min="1"
                            max="5"
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label htmlFor="torusType">Board type:</label>
                        <select id="torusType" value={torusType} onChange={(e) => setTorusType(e.target.value)}>
                            <option value="none">Normal</option>
                            <option value="horizontal">Cilinder</option>
                            <option value="all">Torus</option>
                        </select>
                    </div>
                    <div className="modal-actions">
                        <button type="button" onClick={onClose} className="cancel-button">Cancel</button>
                        <button type="submit" className="submit-button">Send</button>
                    </div>
                </form>
            </div>
        </div>
    );
}

export default MatchRequestModal;
