// client/src/components/SelfMatchRequestModal.js

import React, { useState } from 'react';
import './MatchRequestModal.css';

function SelfMatchRequestModal({onClose, onConfirm}) {
    const [boardSize, setBoardSize] = useState(19);
    //const [timeLimit, setTimeLimit] = useState(10);
    //const [byoyomiTime, setByoyomiTime] = useState(30);
    //const [byoyomiCount, setByoyomiCount] = useState(3);
    const [torusType, setTorusType] = useState('none');

    const handleConfirm = () => {
        onConfirm({
            boardSize: parseInt(boardSize),
            //timeLimit: parseInt(timeLimit),
            //byoyomiTime: parseInt(byoyomiTime),
            //byoyomiCount: parseInt(byoyomiCount),
            torusType: torusType,
        });
    };

    return (
        <div className="modal-overlay">
            <div className="modal-content">
                <h3>Self Play</h3>
                    <div className="form-group">
                        <label htmlFor="boardSize">Board Size:</label>
                        <select id="boardSize" value={boardSize} onChange={(e) => setBoardSize(e.target.value)}>
                            <option value={9}>9×9</option>
                            <option value={13}>13×13</option>
                            <option value={17}>17×17</option>
                            <option value={19}>19×19</option>
                        </select>
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

export default SelfMatchRequestModal;
