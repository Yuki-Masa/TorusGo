import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PIL import Image, ImageDraw

import pygame
import sys

"""Draft"""

class GoBoardRenderer:
    def __init__(self, board_size, cell_size=None, board_padding=None, is_small=False):
        self.board_size = board_size
        self.BOARD_AREA_SIZE = 600
        self.is_small = is_small

        if cell_size is None or board_padding is None:
            self.CELL_SIZE = (self.BOARD_AREA_SIZE - self.BOARD_AREA_SIZE / 10) / ((board_size - 1) * 2) if is_small else (self.BOARD_AREA_SIZE - self.BOARD_AREA_SIZE / 10) / (board_size - 1)
            self.BOARD_PADDING = self.BOARD_AREA_SIZE / 40 if is_small else self.BOARD_AREA_SIZE / 20
        else:
            self.CELL_SIZE = cell_size
            self.BOARD_PADDING = board_padding

        self.board_texture_id = None
        self.torus_radius = 200.0 # トーラスの主半径を拡大
        self.tube_radius = 100.0 # トーラスのチューブ半径を拡大し、縦方向の間隔を調整
        self.detail_x = 60 # トーラスの周囲のセグメント数 (トーラスの形状自体に影響)
        self.detail_y = 30 # チューブに沿ったセグメント数 (トーラスの形状自体に影響)
        self.line_render_detail = 30 # 格子線を描画する際の補間点の数

        self._create_board_texture() # トーラスの「木目」テクスチャを生成 (グリッド線は含めない)

        # マウス操作のための変数
        self.rotate_x = 0.0
        self.rotate_y = 0.0
        self.is_dragging = False
        self.prev_mouse_x = 0
        self.prev_mouse_y = 0

        # ズームのための変数
        self.zoom_level = 1.0
        self.initial_z_translation = 300.0

    def _create_board_texture(self):
        # 碁盤の木目テクスチャを生成 (グリッド線は含めない)
        texture_width = int(self.BOARD_PADDING * 2 + (self.board_size - 1) * self.CELL_SIZE)
        texture_height = int(self.BOARD_PADDING * 2 + (self.board_size - 1) * self.CELL_SIZE)

        img = Image.new('RGB', (texture_width, texture_height), color='#D4A87C') # 碁盤の色
        # この画像にはグリッド線や星点は描画しない。これらは3D空間に直接描画する。

        img_data = img.tobytes("raw", "RGB", 0, 1) # RGB形式でテクスチャデータを取得
        self.board_texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.board_texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
        glBindTexture(GL_TEXTURE_2D, 0)

    def _calculate_torus_point(self, u, v):
        # トーラス上の3D座標を計算
        x = (self.torus_radius + self.tube_radius * np.cos(v)) * np.cos(u)
        y = (self.torus_radius + self.tube_radius * np.cos(v)) * np.sin(u)
        z = self.tube_radius * np.sin(v)
        return x, y, z

    def _draw_disk(self, radius, segments=30):
        # 平面的な円盤を描画する
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0.0, 0.0, 0.0) # 円盤の中心
        for i in range(segments + 1):
            angle = 2 * np.pi * i / segments
            glVertex3f(radius * np.cos(angle), radius * np.sin(angle), 0.0)
        glEnd()

    def handle_mouse_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1: # 左クリック
                self.is_dragging = True
                self.prev_mouse_x, self.prev_mouse_y = event.pos
            elif event.button == 4: # マウスホイールアップ (ズームイン)
                self.zoom_level = max(0.2, self.zoom_level - 0.1) # 最小ズームレベルを0.2に設定
            elif event.button == 5: # マウスホイールダウン (ズームアウト)
                self.zoom_level = min(3.0, self.zoom_level + 0.1) # 最大ズームレベルを3.0に設定
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1: # 左クリック解除
                self.is_dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.is_dragging:
                dx = event.pos[0] - self.prev_mouse_x
                dy = event.pos[1] - self.prev_mouse_y
                self.rotate_y += dx * 0.5 # Y軸周りの回転
                self.rotate_x += dy * 0.5 # X軸周りの回転
                self.prev_mouse_x, self.prev_mouse_y = event.pos

    def draw_torus_board(self, board_state, current_offset=(0, 0)):
        offset_x, offset_y = current_offset

        glPushMatrix()
        glTranslatef(0, 0, -self.initial_z_translation * self.zoom_level) # ズームレベルを適用
        glRotatef(self.rotate_x, 1, 0, 0) # マウス操作による回転を適用
        glRotatef(self.rotate_y, 0, 1, 0)

        # トーラス表面の描画 (半透明)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glDisable(GL_LIGHTING) # トーラス描画中はライティングを無効化

        # トーラスの色を半透明の白に近い水色に設定
        glColor4f(0.8, 0.9, 1.0, 0.7) # 半透明の薄い水色

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.board_texture_id)
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE) # ここでGL_MODULATEを明示的に設定

        glEnable(GL_POLYGON_OFFSET_FILL) # Zファイティング対策: ポリゴンオフセットを有効化
        glPolygonOffset(1.0, 1.0) # オフセット量を設定

        glBegin(GL_QUADS)
        for i in range(self.detail_x):
            u = np.interp(i, [0, self.detail_x], [0, 2 * np.pi])
            u_next = np.interp(i + 1, [0, self.detail_x], [0, 2 * np.pi])

            for j in range(self.detail_y):
                v = np.interp(j, [0, self.detail_y], [0, 2 * np.pi])
                v_next = np.interp(j + 1, [0, self.detail_y], [0, 2 * np.pi])

                tx_u = np.interp(u, [0, 2 * np.pi], [0, 1])
                tx_u_next = np.interp(u_next, [0, 2 * np.pi], [0, 1])
                tx_v = np.interp(v, [0, 2 * np.pi], [0, 1])
                tx_v_next = np.interp(v_next, [0, 2 * np.pi], [0, 1])

                glTexCoord2f(tx_u, tx_v)
                glVertex3f(*self._calculate_torus_point(u, v))
                glTexCoord2f(tx_u_next, tx_v)
                glVertex3f(*self._calculate_torus_point(u_next, v))
                glTexCoord2f(tx_u_next, tx_v_next)
                glVertex3f(*self._calculate_torus_point(u_next, v_next))
                glTexCoord2f(tx_u, tx_v_next)
                glVertex3f(*self._calculate_torus_point(u, v_next))
        glEnd()
        glDisable(GL_POLYGON_OFFSET_FILL) # ポリゴンオフセットを無効化

        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING) # トーラス描画後、ライティングを再度有効化

        # 碁盤の格子線と星点を直接描画 (黒色)
        glColor3f(0.0, 0.0, 0.0) # 黒色
        glLineWidth(1.5) # 線幅を少し太くする (好みで調整)

        # 水平線 (行)
        for r_line in range(self.board_size):
            glBegin(GL_LINE_STRIP)
            # 各グリッド単位で line_render_detail 個の点を補間
            for i in range((self.board_size - 1) * self.line_render_detail + 1):
                # current_c_logical は0からboard_size-1まで滑らかに変化
                current_c_logical = np.interp(i, [0, (self.board_size - 1) * self.line_render_detail], [0, self.board_size - 1])

                wrapped_r = (r_line - offset_y) % self.board_size
                wrapped_c = (current_c_logical - offset_x) % self.board_size

                u_val = np.interp(wrapped_c, [0, self.board_size - 1], [0, 2 * np.pi])
                # vの範囲を[0, 2*pi]に戻す
                v_val = np.interp(wrapped_r, [0, self.board_size - 1], [0, 2 * np.pi])
                glVertex3f(*self._calculate_torus_point(u_val, v_val))
            glEnd()

        # 垂直線 (列)
        for c_line in range(self.board_size):
            glBegin(GL_LINE_STRIP)
            # 各グリッド単位で line_render_detail 個の点を補間
            for i in range((self.board_size - 1) * self.line_render_detail + 1):
                # current_r_logical は0からboard_size-1まで滑らかに変化
                current_r_logical = np.interp(i, [0, (self.board_size - 1) * self.line_render_detail], [0, self.board_size - 1])

                wrapped_r = (current_r_logical - offset_y) % self.board_size
                wrapped_c = (c_line - offset_x) % self.board_size

                u_val = np.interp(wrapped_c, [0, self.board_size - 1], [0, 2 * np.pi])
                # vの範囲を[0, 2*pi]に戻す
                v_val = np.interp(wrapped_r, [0, self.board_size - 1], [0, 2 * np.pi])
                glVertex3f(*self._calculate_torus_point(u_val, v_val))
            glEnd()

        # # 星点の描画
        # star_points = []
        # if self.board_size == 9:
        #     star_points = [(2, 2), (2, 6), (6, 2), (6, 6), (4, 4)]
        # elif self.board_size == 13:
        #     star_points = [(3, 3), (3, 9), (9, 3), (9, 9), (6, 6)]
        # elif self.board_size == 19:
        #     star_points = [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]

        # for r_star, c_star in star_points:
        #     wrapped_r = (r_star - offset_y) % self.board_size
        #     wrapped_c = (c_star - offset_x) % self.board_size

        #     u_star = np.interp(wrapped_c, [0, self.board_size - 1], [0, 2 * np.pi])
        #     # vの範囲を[0, 2*pi]に戻す
        #     v_star = np.interp(wrapped_r, [0, self.board_size - 1], [0, 2 * np.pi])

        #     x_s, y_s, z_s = self._calculate_torus_point(u_star, v_star)

        #     glPushMatrix()
        #     glTranslatef(x_s, y_s, z_s)

        #     # トーラスのパラメータ表示の接線ベクトルを計算
        #     tan_u_vec = np.array([
        #         -(self.torus_radius + self.tube_radius * np.cos(v_star)) * np.sin(u_star),
        #         (self.torus_radius + self.tube_radius * np.cos(v_star)) * np.cos(u_star),
        #         0
        #     ])
        #     tan_v_vec = np.array([
        #         -self.tube_radius * np.sin(v_star) * np.cos(u_star),
        #         -self.tube_radius * np.sin(v_star) * np.sin(u_star),
        #         self.tube_radius * np.cos(v_star)
        #     ])

        #     # 法線ベクトルは接線ベクトルの外積
        #     normal_vec = np.cross(tan_u_vec, tan_v_vec)
        #     # ゼロ除算を避けて正規化
        #     normal_vec_norm = np.linalg.norm(normal_vec)
        #     normal_vec = normal_vec / normal_vec_norm if normal_vec_norm != 0 else np.array([0, 0, 1])

        #     # ローカルX軸 (right_vec) をT_uから取得し正規化
        #     right_vec_norm = np.linalg.norm(tan_u_vec)
        #     right_vec = tan_u_vec / right_vec_norm if right_vec_norm != 0 else np.array([1, 0, 0])

        #     # ローカルY軸 (new_up_vec) を法線とローカルX軸の外積から計算し正規化
        #     new_up_vec = np.cross(normal_vec, right_vec)
        #     new_up_vec_norm = np.linalg.norm(new_up_vec)
        #     new_up_vec = new_up_vec / new_up_vec_norm if new_up_vec_norm != 0 else np.array([0, 1, 0])

        #     # 厳密な直交性を保証するため、right_vecを再計算 (GSプロセス)
        #     right_vec = np.cross(new_up_vec, normal_vec)
        #     right_vec_norm_recalc = np.linalg.norm(right_vec)
        #     right_vec = right_vec / right_vec_norm_recalc if right_vec_norm_recalc != 0 else np.array([1, 0, 0])

        #     rot_matrix = np.array([
        #         right_vec[0], new_up_vec[0], normal_vec[0], 0,
        #         right_vec[1], new_up_vec[1], normal_vec[1], 0,
        #         right_vec[2], new_up_vec[2], normal_vec[2], 0,
        #         0, 0, 0, 1
        #     ], dtype=np.float32)
        #     glMultMatrixf(rot_matrix)

        #     # 星点を小さな円盤として描画 (Zファイティング対策のため少し持ち上げる)
        #     glTranslatef(0.0, 0.0, 0.0001) # 非常に小さい値に設定
        #     glColor3f(0.0, 0.0, 0.0) # 黒色
        #     self._draw_disk(self.CELL_SIZE * 0.1) # 星点の半径をCELL_SIZEの10%に (好みで調整)
        #     glPopMatrix()


        # 石の描画 (平面ディスク)
        stone_radius = self.CELL_SIZE * 0.4 # 石の半径を調整

        for r_idx in range(self.board_size):
            for c_idx in range(self.board_size):
                stone = board_state[r_idx][c_idx]

                if stone != 0:
                    wrapped_r = (r_idx - offset_y) % self.board_size
                    wrapped_c = (c_idx - offset_x) % self.board_size

                    u_stone = np.interp(wrapped_c, [0, self.board_size - 1], [0, 2 * np.pi])
                    # vの範囲を[0, 2*pi]に戻す
                    v_stone = np.interp(wrapped_r, [0, self.board_size - 1], [0, 2 * np.pi])

                    x_s, y_s, z_s = self._calculate_torus_point(u_stone, v_stone)

                    glPushMatrix()
                    glTranslatef(x_s, y_s, z_s)

                    # 石の向きを碁盤の表面に合わせるための計算を削除（常にワールドXY平面に水平）
                    # tan_u_vec = np.array([
                    #     -(self.torus_radius + self.tube_radius * np.cos(v_stone)) * np.sin(u_stone),
                    #     (self.torus_radius + self.tube_radius * np.cos(v_stone)) * np.cos(u_stone),
                    #     0
                    # ])
                    # tan_v_vec = np.array([
                    #     -self.tube_radius * np.sin(v_stone) * np.cos(u_stone),
                    #     -self.tube_radius * np.sin(v_stone) * np.sin(u_stone),
                    #     self.tube_radius * np.cos(v_stone)
                    # ])

                    # normal_vec = np.cross(tan_u_vec, tan_v_vec)
                    # normal_vec_norm = np.linalg.norm(normal_vec)
                    # normal_vec = normal_vec / normal_vec_norm if normal_vec_norm != 0 else np.array([0, 0, 1])

                    # right_vec_norm = np.linalg.norm(tan_u_vec)
                    # right_vec = tan_u_vec / right_vec_norm if right_vec_norm != 0 else np.array([1, 0, 0])

                    # new_up_vec = np.cross(normal_vec, right_vec)
                    # new_up_vec_norm = np.linalg.norm(new_up_vec)
                    # new_up_vec = new_up_vec / new_up_vec_norm if new_up_vec_norm != 0 else np.array([0, 1, 0])

                    # right_vec = np.cross(new_up_vec, normal_vec)
                    # right_vec_norm_recalc = np.linalg.norm(right_vec)
                    # right_vec = right_vec / right_vec_norm_recalc if right_vec_norm_recalc != 0 else np.array([1, 0, 0])

                    # rot_matrix = np.array([
                    #     right_vec[0], new_up_vec[0], normal_vec[0], 0,
                    #     right_vec[1], new_up_vec[1], normal_vec[1], 0,
                    #     right_vec[2], new_up_vec[2], normal_vec[2], 0,
                    #     0, 0, 0, 1
                    # ], dtype=np.float32)
                    # glMultMatrixf(rot_matrix) # 回転行列の適用を削除

                    # 石をトーラス表面から非常にわずかに持ち上げる (Zファイティング対策)
                    glTranslatef(0.0, 0.0, 0.0001)

                    if stone == 1:
                        glColor3f(0.0, 0.0, 0.0) # 黒石
                    elif stone == 2:
                        glColor3f(1.0, 1.0, 1.0) # 白石

                    self._draw_disk(stone_radius) # 石を平面円盤として描画

                    glPopMatrix()
        glPopMatrix()

# --- OpenGLの初期化と描画ループ ---

WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
BOARD_SIZE = 9 # 碁盤サイズを9x9に変更
INITIAL_OFFSET = (0, 0)

dummy_board_state = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
# ダミーの碁盤状態をいくつか設定
dummy_board_state[2][2] = 1
dummy_board_state[2][3] = 2
dummy_board_state[3][2] = 2
dummy_board_state[4][4] = 1
dummy_board_state[0][0] = 1
dummy_board_state[BOARD_SIZE-1][BOARD_SIZE-1] = 2
dummy_board_state[0][BOARD_SIZE-1] = 1
dummy_board_state[BOARD_SIZE-1][0] = 2


renderer = None

def init_gl():
    glClearColor(0.7, 0.7, 0.7, 1.0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, (0, 0, 1, 0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1.0))
    glLightfv(GL_LIGHT0, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (WINDOW_WIDTH / WINDOW_HEIGHT), 0.1, 1000.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(0, 0, 500, 0, 0, 0, 0, 1, 0)

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluLookAt(0, 0, 500, 0, 0, 0, 0, 1, 0)

    global renderer
    renderer.draw_torus_board(dummy_board_state, current_offset=INITIAL_OFFSET)

    pygame.display.flip()

def main():
    pygame.init()
    glutInit(sys.argv)
    pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("Torus Go Board (PyOpenGL)")

    init_gl()
    global renderer
    renderer = GoBoardRenderer(BOARD_SIZE)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            renderer.handle_mouse_event(event)

        display()
        pygame.time.wait(10)

    pygame.quit()

if __name__ == '__main__':
    main()
