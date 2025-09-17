// client/src/components/GoBoardComponent.js

import React, { useRef, useEffect, useCallback } from 'react';
import './GoBoardComponent.css';

//export const BOARD_PADDING = 25; // 碁盤の端からの余白
export const BOARD_AREA_SIZE = 600;

function GoBoardComponent({ boardState, boardSize, isSmall, torusType, onPlaceStone, isMyTurn, offset = { x: 0, y: 0 }, isClickable = true }) {
    const canvasRef = useRef(null);
    // isSmallによってCELL_SIZEが変わるように修正
    //const CELL_SIZE = isSmall ? (BOARD_AREA_SIZE - BOARD_PADDING * 2) / ((boardSize - 1) * 2) : (BOARD_AREA_SIZE - BOARD_PADDING * 2) / (boardSize - 1);
    const CELL_SIZE = isSmall ? (BOARD_AREA_SIZE - BOARD_AREA_SIZE/10) / ((boardSize - 1) * 2) : (BOARD_AREA_SIZE - BOARD_AREA_SIZE/10) / (boardSize - 1);
    const BOARD_PADDING = isSmall ? BOARD_AREA_SIZE/40:BOARD_AREA_SIZE/20;

    // 座標変換ヘルパー
    const _wrapCoord = useCallback((coord, max) => {
        return (coord % max + max) % max;
    }, []);

    // 碁盤と石の描画ロジック
    const drawBoard = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        // canvasのサイズをisSmallに応じて設定
        canvas.width = isSmall ? (BOARD_AREA_SIZE / 2) : BOARD_AREA_SIZE;
        canvas.height = isSmall ? (BOARD_AREA_SIZE / 2) : BOARD_AREA_SIZE;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        ctx.fillStyle = '#D4A87C'; // 碁盤の色
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.strokeStyle = '#000';
        ctx.lineWidth = 1;

        // グリッド線と星の描画はオフセットと関係なく、碁盤の標準的な位置に描画
        // 行と列の描画
        for (let i = 0; i < boardSize; i++) {
            // 水平線
            ctx.beginPath();
            ctx.moveTo(BOARD_PADDING, BOARD_PADDING + i * CELL_SIZE);
            ctx.lineTo(BOARD_PADDING + (boardSize - 1) * CELL_SIZE, BOARD_PADDING + i * CELL_SIZE);
            ctx.stroke();

            // 垂直線
            ctx.beginPath();
            ctx.moveTo(BOARD_PADDING + i * CELL_SIZE, BOARD_PADDING);
            ctx.lineTo(BOARD_PADDING + i * CELL_SIZE, BOARD_PADDING + (boardSize - 1) * CELL_SIZE);
            ctx.stroke();
        }

        // 星を描画
        const starPoints = [];
        if (boardSize === 9) {
            starPoints.push([2, 2], [2, 6], [6, 2], [6, 6], [4, 4]);
        } else if (boardSize === 13) {
            starPoints.push([3, 3], [3, 9], [9, 3], [9, 9], [6, 6]);
        } else if (boardSize === 19) {
            starPoints.push([3, 3], [3, 9], [3, 15], [9, 3], [9, 9], [9, 15], [15, 3], [15, 9], [15, 15]);
        }
        ctx.fillStyle = '#000';
        starPoints.forEach(([r, c]) => {
            const x = BOARD_PADDING + c * CELL_SIZE;
            const y = BOARD_PADDING + r * CELL_SIZE;
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, 2 * Math.PI);
            ctx.fill();
        });

        // 石を描画 (オフセットとトーラスを考慮)
        // 画面に表示される範囲を計算
        const visibleWidth = canvas.width - BOARD_PADDING * 2;
        const visibleHeight = canvas.height - BOARD_PADDING * 2;

        //(ここ必要なし)オフセットがCELL_SIZEの倍数でない場合、部分的に表示される石を考慮するため、描画範囲を少し広げる
        //const renderMargin = 2; // 表示範囲の端にある石を描画するためのマージン (石の半径分)
        /*
        const startRenderX = Math.floor((-offset.x - renderMargin) / CELL_SIZE);
        const endRenderX = Math.ceil((visibleWidth - offset.x + renderMargin) / CELL_SIZE);
        const startRenderY = Math.floor((-offset.y - renderMargin) / CELL_SIZE);
        const endRenderY = Math.ceil((visibleHeight - offset.y + renderMargin) / CELL_SIZE);
        */
        const startRenderX = Math.floor((-offset.x*CELL_SIZE) / CELL_SIZE);
        const endRenderX = Math.ceil((visibleWidth - offset.x*CELL_SIZE ) / CELL_SIZE);
        const startRenderY = Math.floor((-offset.y*CELL_SIZE) / CELL_SIZE);
        const endRenderY = Math.ceil((visibleHeight - offset.y*CELL_SIZE) / CELL_SIZE);


        for (let r = startRenderY; r <= endRenderY; r++) {
            for (let c = startRenderX; c <= endRenderX; c++) {
                let actualR = r;
                let actualC = c;

                // トーラスタイプに応じて座標をラップ
                if (torusType === 'all') {
                    actualR = _wrapCoord(r, boardSize);
                    actualC = _wrapCoord(c, boardSize);
                } else if (torusType === 'horizontal') {
                    actualC = _wrapCoord(c, boardSize);
                    // 行方向はラップしないので、範囲外ならスキップ
                    if (actualR < 0 || actualR >= boardSize) continue;
                }
                // 'none'の場合はそのまま

                // ボードの状態から石を取得
                const stone = boardState[actualR][actualC];

                if (stone !== 0) {
                    // 描画位置は、元のグリッド座標にオフセットを適用
                    /*
                    const x = BOARD_PADDING + c * CELL_SIZE + offset.x;
                    const y = BOARD_PADDING + r * CELL_SIZE + offset.y;
                    */
                    const x = BOARD_PADDING + c * CELL_SIZE + offset.x*CELL_SIZE;
                    const y = BOARD_PADDING + r * CELL_SIZE + offset.y*CELL_SIZE;
                  　//alert(isSmall+(x-BOARD_PADDING/CELL_SIZE)%boardSize+","+(y-BOARD_PADDING/CELL_SIZE));
                    ctx.beginPath();
                    //ctx.arc(x, y, CELL_SIZE / 2 - 2, 0, 2 * Math.PI); // 石の半径はCELL_SIZEの半分から少し小さく
                    ctx.arc(x, y, CELL_SIZE / 2 , 0, 2 * Math.PI);

                    if (stone === 1) {
                        ctx.fillStyle = '#000'; // 黒石
                        ctx.strokeStyle = '#000';
                    } else if (stone === 2) {
                        ctx.fillStyle = '#fff'; // 白石
                        ctx.strokeStyle = '#000';
                    }
                    ctx.fill();
                    ctx.stroke();
                }
            }
        }

    }, [boardState, boardSize, offset, _wrapCoord, CELL_SIZE, isSmall, torusType]); // 依存配列にCELL_SIZE, isSmall, torusTypeを追加

    useEffect(() => {
        drawBoard();
    }, [drawBoard]);

    const handleClick = (e) => {
        if (!isClickable || !isMyTurn) return;

        const canvas = canvasRef.current;
        const rect = canvas.getBoundingClientRect();

        const clientX = e.clientX || e.touches[0].clientX;
        const clientY = e.clientY || e.touches[0].clientY;

        // クリックされた位置からキャンバス内の相対座標を計算
        // そしてオフセットを考慮して「表示されている碁盤上の座標」に変換
        //const xInCanvas = clientX - rect.left - BOARD_PADDING - offset.x;
        //const yInCanvas = clientY - rect.top - BOARD_PADDING - offset.y;
        const xInCanvas = clientX - rect.left - BOARD_PADDING;
        const yInCanvas = clientY - rect.top - BOARD_PADDING;

        console.log(`offset:`,offset);

        // 碁盤のグリッド座標に変換 (トーラス構造はここで考慮しない、サーバーサイドで処理)
        // Math.round を使うことで、交点に近い場所をクリックした場合に正しくその交点が選ばれる
        //let col = Math.round(xInCanvas / CELL_SIZE)%boardSize;
        //let row = Math.round(yInCanvas / CELL_SIZE)%boardSize;
        let col = Math.round(xInCanvas / CELL_SIZE)%boardSize-(offset.x%boardSize);
        let row = Math.round(yInCanvas / CELL_SIZE)%boardSize-(offset.y%boardSize);
        //alert("coord"+col+","+row);

        col = Math.max(col, (boardSize+col)%boardSize);
        row = Math.max(row, (boardSize+row)%boardSize);
        //alert("coord"+col+","+row);


        // 0からboardSize-1の範囲に丸める（トーラス構造はサーバーで最終的にラップされるが、
        // クライアント側で表示されている範囲内のグリッドであることを確認するため）
        //col = Math.max(0, Math.min(boardSize - 1, col));
        //row = Math.max(0, Math.min(boardSize - 1, row));

        //alert('offset'+offset.x+','+offset.y);

        onPlaceStone(row, col);
    };

    return (
        <canvas
            ref={canvasRef}
            className={`go-board-canvas ${isClickable ? 'clickable' : ''} ${isMyTurn && isClickable ? 'my-turn' : ''}`}
            onClick={handleClick}
            onTouchStart={handleClick}
        />
    );
}

export default GoBoardComponent;
