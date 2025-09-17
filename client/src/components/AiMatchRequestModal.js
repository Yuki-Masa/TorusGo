// client/src/components/AiMatchRequestModal.js

import React, { useState } from 'react';
import './MatchRequestModal.css';

function AiMatchRequestModal({onClose, onConfirm}) {
    const [boardSize, setBoardSize] = useState(6);
    const [timeLimit, setTimeLimit] = useState(10);
    const [byoyomiTime, setByoyomiTime] = useState(30);
    const [byoyomiCount, setByoyomiCount] = useState(3);
    const [torusType, setTorusType] = useState('none'); // トーラスタイプを追加

    const handleConfirm = () => {
        onConfirm({
            boardSize: parseInt(boardSize),
            timeLimit: parseInt(timeLimit),
            byoyomiTime: parseInt(byoyomiTime),
            byoyomiCount: parseInt(byoyomiCount),
            torusType: torusType,
        });
    };

    return (
        <div className="modal-overlay">
            <div className="modal-content">
                <h3>NPC Match</h3>
                    <div className="form-group">
                        <label htmlFor="boardSize">Board Size:</label>
                        <select id="boardSize" value={boardSize} onChange={(e) => setBoardSize(e.target.value)}>
                            <option value={6}>6×6</option>
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
                        <label htmlFor="torusType">Board Type:</label>
                        <select id="torusType" value={torusType} onChange={(e) => setTorusType(e.target.value)}>
                            <option value="none">Normal</option>
                            <option value="horizontal">Cilinder</option>
                            <option value="all">Torus</option>
                        </select>
                    </div>
                    <div className="modal-actions">
                        <button onClick={onClose} className="cancel-button">Cancel</button>
                        <button onClick={handleConfirm}>Start Game</button>
                    </div>
            </div>
        </div>
    );
}

export default AiMatchRequestModal;
