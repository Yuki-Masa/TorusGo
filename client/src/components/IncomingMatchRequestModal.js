// client/src/components/IncomingMatchRequestModal.js

import React, { useState, useEffect } from 'react';
import './IncomingMatchRequestModal.css';

function IncomingMatchRequestModal({ requestDetails, onConfirm, onCancel, onCorrect }) {
    const [selectedBoardSize, setSelectedBoardSize] = useState(requestDetails?.boardSize || 19);
    const [selectedTimeLimit, setSelectedTimeLimit] = useState(requestDetails?.timeLimit || 10);
    const [selectedByoyomiTime, setSelectedByoyomiTime] = useState(requestDetails?.byoyomiTime || 30);
    const [selectedByoyomiCount, setSelectedByoyomiCount] = useState(requestDetails?.byoyomiCount || 3);
    const [selectedTorusType, setSelectedTorusType] = useState(requestDetails?.torusType || 'none');

    useEffect(() => {
        if (requestDetails) {
            setSelectedBoardSize(requestDetails.boardSize || 19);
            setSelectedTimeLimit(requestDetails.timeLimit || 10);
            setSelectedByoyomiTime(requestDetails.byoyomiTime || 30);
            setSelectedByoyomiCount(requestDetails.byoyomiCount || 3);
            setSelectedTorusType(requestDetails.torusType || 'none');
        }
    }, [requestDetails]);

    useEffect(() => {
        document.body.style.overflow = 'hidden';
        return () => {
            document.body.style.overflow = 'unset';
        };
    }, []);

    if (!requestDetails) {
        console.log('[IncomingMatchRequestModal] requestDetails is null, not rendering.');
        return null;
    }

    const modalTitle = requestDetails.isCorrection ? "修正申し込み" : "対局申し込み";
    const message = `
        ${requestDetails.requesterUsername}さんから${requestDetails.isCorrection ? "修正申し込み" : "対局申し込み"}がありました。<br>
        現在の条件を確認・修正して返信してください。
    `;

    const handleConfirm = () => {
        onConfirm({
            boardSize: parseInt(selectedBoardSize),
            timeLimit: parseInt(selectedTimeLimit),
            byoyomiTime: parseInt(selectedByoyomiTime),
            byoyomiCount: parseInt(selectedByoyomiCount),
            torusType: selectedTorusType,
            modifiedSettings: {
                boardSize: parseInt(selectedBoardSize),
                timeLimit: parseInt(selectedTimeLimit),
                byoyomiTime: parseInt(selectedByoyomiTime),
                byoyomiCount: parseInt(selectedByoyomiCount),
                torusType: selectedTorusType
            },
            initialSettings: {
                boardSize: parseInt(selectedBoardSize),
                timeLimit: parseInt(selectedTimeLimit),
                byoyomiTime: parseInt(selectedByoyomiTime),
                byoyomiCount: parseInt(selectedByoyomiCount),
                torusType: selectedTorusType
            }
        });
    };

    const handleCancel = () => {
        onCancel({});
    };

    const handleCorrect = () => {
        onCorrect({
            modifiedSettings: {
                boardSize: parseInt(selectedBoardSize),
                timeLimit: parseInt(selectedTimeLimit),
                byoyomiTime: parseInt(selectedByoyomiTime),
                byoyomiCount: parseInt(selectedByoyomiCount),
                torusType: selectedTorusType
            },
            initialSettings: {
                boardSize: requestDetails.boardSize,
                timeLimit: requestDetails.timeLimit,
                byoyomiTime: requestDetails.byoyomiTime,
                byoyomiCount: requestDetails.byoyomiCount,
                torusType: requestDetails.torusType
            },
            isCorrection: true
        });
    };

    return (
        <div className="modal-backdrop">
            <div className="modal-content">
                <h3>{modalTitle}</h3>
                <p dangerouslySetInnerHTML={{ __html: message }}></p>

                <div className="form-group">
                    <label htmlFor="boardSize">ボードサイズ:</label>
                    <select
                        id="boardSize"
                        value={selectedBoardSize}
                        onChange={(e) => setSelectedBoardSize(e.target.value)}
                    >
                        <option value={9}>9路盤</option>
                        <option value={13}>13路盤</option>
                        <option value={19}>19路盤</option>
                    </select>
                </div>
                <div className="form-group">
                    <label htmlFor="timeLimit">持ち時間 (分):</label>
                    <input
                        type="number"
                        id="timeLimit"
                        value={selectedTimeLimit}
                        onChange={(e) => setSelectedTimeLimit(e.target.value)}
                        min="0"
                        max="60"
                        required
                    />
                </div>
                <div className="form-group">
                    <label htmlFor="byoyomiTime">秒読み時間 (秒):</label>
                    <input
                        type="number"
                        id="byoyomiTime"
                        value={selectedByoyomiTime}
                        onChange={(e) => setSelectedByoyomiTime(e.target.value)}
                        min="10"
                        max="60"
                        required
                    />
                </div>
                <div className="form-group">
                    <label htmlFor="byoyomiCount">秒読み回数:</label>
                    <input
                        type="number"
                        id="byoyomiCount"
                        value={selectedByoyomiCount}
                        onChange={(e) => setSelectedByoyomiCount(e.target.value)}
                        min="1"
                        max="5"
                        required
                    />
                </div>

                <div className="form-group">
                    <label htmlFor="torusType">トーラスタイプ:</label>
                    <select
                        id="torusType"
                        value={selectedTorusType}
                        onChange={(e) => setSelectedTorusType(e.target.value)}
                    >
                        <option value="none">通常</option>
                        <option value="horizontal">左右トーラス</option>
                        <option value="all">上下左右トーラス</option>
                    </select>
                </div>

                <div className="modal-actions">
                    <button onClick={handleCancel} className="deny-button">拒否</button>
                    <button onClick={handleCorrect} className="correct-button">修正申し込み</button>
                    <button onClick={handleConfirm} className="accept-button">承諾</button>
                </div>
            </div>
        </div>
    );
}

export default IncomingMatchRequestModal;
