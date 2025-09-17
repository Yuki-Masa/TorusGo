// client/src/components/EndGameModal.js

import React from 'react';
import './EndGameModal.css';

function EndGameModal({ message, onConfirm, onCancel, confirmText, cancelText }) {
    return (
        <div className="modal-overlay">
            <div className="modal-content">
                <p className="modal-message" dangerouslySetInnerHTML={{ __html: message.replace(/\n/g, '<br>') }} />
                <div className="modal-actions">
                    <button onClick={onCancel} className="cancel-button">{cancelText}</button>
                    <button onClick={onConfirm} className="confirm-button">{confirmText}</button>
                </div>
            </div>
        </div>
    );
}

export default EndGameModal;
