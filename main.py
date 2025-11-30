import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
from matplotlib.animation import FuncAnimation
import platform

# ---------------------------------------------------------
# 0. Mac用フォント設定
# ---------------------------------------------------------
system = platform.system()
if system == 'Darwin':  # Mac
    plt.rcParams['font.family'] = 'Hiragino Sans'
else:
    plt.rcParams['font.family'] = 'sans-serif'

# ---------------------------------------------------------
# 1. データの準備
# ---------------------------------------------------------
y = np.array([
    [0.1, 0.2, 0.6, 0.1],
    [0.8, 0.1, 0.0, 0.1],
    [0.3, 0.3, 0.1, 0.3]
])
t = np.array([2, 0, 1])
batch_size = y.shape[0]
num_classes = y.shape[1]
row_indices = np.arange(batch_size)
extracted_values = y[row_indices, t]

# ---------------------------------------------------------
# 2. ビジュアル設定
# ---------------------------------------------------------
fig = plt.figure(figsize=(15, 7), facecolor='#F0F2F5')
gs = fig.add_gridspec(1, 4, width_ratios=[4, 1.5, 1.5, 1.5], wspace=0.6, left=0.05, right=0.95)

ax_y = fig.add_subplot(gs[0])
ax_rows = fig.add_subplot(gs[1])
ax_t = fig.add_subplot(gs[2])
ax_ext = fig.add_subplot(gs[3])

for ax in [ax_y, ax_rows, ax_t, ax_ext]:
    ax.set_facecolor('#F0F2F5')
    ax.set_axis_off()

# ---------------------------------------------------------
# 3. 描画用ヘルパー関数
# ---------------------------------------------------------
def draw_base_matrix(ax, data, title):
    rows, cols = data.shape
    ax.set_xlim(-0.5, cols + 0.5)
    ax.set_ylim(-0.5, rows + 0.5)
    ax.text(cols/2, rows + 0.6, title, ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    for c in range(cols):
        ax.text(c + 0.5, rows + 0.1, f"Cls {c}", ha='center', va='bottom', fontsize=9)
    for r in range(rows):
        ax.text(-0.2, rows - 1 - r + 0.5, f"Data {r}", ha='right', va='center', fontsize=9)

    for r in range(rows):
        for c in range(cols):
            y_pos = rows - 1 - r
            ax.text(c + 0.5, y_pos + 0.5, f"{data[r, c]:.1f}", 
                    ha='center', va='center', fontsize=14, color='gray', zorder=5)
            rect = patches.Rectangle((c, y_pos), 1, 1, linewidth=1, edgecolor='#CCCCCC', facecolor='white', zorder=1)
            ax.add_patch(rect)

def draw_base_vector(ax, data, title, bg_header_color):
    rows = len(data)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, rows)
    ax.text(0.5, rows + 0.2, title, ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    box = patches.Rectangle((0, 0), 1, rows, linewidth=2, edgecolor='gray', facecolor=bg_header_color, zorder=0, alpha=0.3)
    ax.add_patch(box)

    for r in range(rows):
        y_pos = rows - 1 - r
        val = data[r]
        txt = f"{val:.1f}" if isinstance(val, float) else str(val)
        ax.text(0.5, y_pos + 0.5, txt, ha='center', va='center', fontsize=14, color='gray', zorder=5)
        if r < rows - 1:
             ax.plot([0, 1], [y_pos, y_pos], color='gray', linewidth=1, alpha=0.5)

# --- ベースの描画実行 ---
draw_base_matrix(ax_y, y, "y (Output Matrix)")
draw_base_vector(ax_rows, row_indices, "Row Idx\n(arange)", '#D0E0F0')
draw_base_vector(ax_t, t, "Col Idx\n(Target)", '#FDE0C2')
draw_base_vector(ax_ext, extracted_values, "Result\n(Extracted)", '#E0E0E0')

# --- 矢印 ---
arrow_props = {'arrowstyle': '-|>', 'lw': 2, 'color': '#888888', 'mutation_scale': 15}
for i in range(batch_size):
    y_c = batch_size - 1 - i + 0.5
    fig.add_artist(ConnectionPatch(xyA=(0, y_c), xyB=(1, y_c), coordsA="data", coordsB="data", 
                                   axesA=ax_t, axesB=ax_rows, **arrow_props))
    fig.add_artist(ConnectionPatch(xyA=(0, y_c), xyB=(num_classes, y_c), coordsA="data", coordsB="data", 
                                   axesA=ax_rows, axesB=ax_y, **arrow_props))
    fig.add_artist(ConnectionPatch(xyA=(1, y_c), xyB=(0, y_c), coordsA="data", coordsB="data", 
                                   axesA=ax_t, axesB=ax_ext, **arrow_props))

# ---------------------------------------------------------
# 4. アニメーション用パーツ
# ---------------------------------------------------------
hl_style = {'linewidth': 3, 'edgecolor': '#F7931E', 'facecolor': '#FDE0C2', 'alpha': 0.7, 'zorder': 2}
hl_y = patches.Rectangle((0, 0), 1, 1, **hl_style, visible=False)
hl_row = patches.Rectangle((0, 0), 1, 1, **hl_style, visible=False)
hl_t = patches.Rectangle((0, 0), 1, 1, **hl_style, visible=False)
hl_ext = patches.Rectangle((0, 0), 1, 1, **hl_style, visible=False)

ax_y.add_patch(hl_y)
ax_rows.add_patch(hl_row)
ax_t.add_patch(hl_t)
ax_ext.add_patch(hl_ext)

info_text = fig.text(0.5, 0.05, "Ready", ha='center', fontsize=16, color='#0072BC', fontweight='bold')
# クリック操作の説明
fig.text(0.02, 0.02, "Click to Pause/Resume", fontsize=10, color='gray')

# ---------------------------------------------------------
# 5. 更新関数
# ---------------------------------------------------------
def update(frame):
    idx = frame % (batch_size + 2) 

    if idx < batch_size:
        r_idx = row_indices[idx]
        c_idx = t[idx]
        y_pos = batch_size - 1 - idx
        
        hl_row.set_xy((0, y_pos)); hl_row.set_visible(True)
        hl_t.set_xy((0, y_pos)); hl_t.set_visible(True)
        hl_y.set_xy((c_idx, y_pos)); hl_y.set_visible(True)
        hl_ext.set_xy((0, y_pos)); hl_ext.set_visible(True)
        
        info_text.set_text(f"Step {idx+1}:  Row={r_idx}, Col={c_idx}  -->  Extract {y[r_idx, c_idx]}")
        
    else:
        hl_y.set_visible(False)
        hl_row.set_visible(False)
        hl_t.set_visible(False)
        hl_ext.set_visible(False)
        info_text.set_text(f"Result Extracted: {extracted_values}")

# ---------------------------------------------------------
# 6. アニメーション制御と実行
# ---------------------------------------------------------
ani = FuncAnimation(fig, update, frames=(batch_size + 2) * 2, interval=1200, repeat=True)

# 一時停止のフラグ
is_paused = False

def on_click(event):
    global is_paused
    if is_paused:
        ani.event_source.start()
    else:
        ani.event_source.stop()
    is_paused = not is_paused

# クリックイベントを接続
fig.canvas.mpl_connect('button_press_event', on_click)

if __name__ == "__main__":
    # タイトルは削除しました
    plt.subplots_adjust(top=0.9, bottom=0.15)
    plt.show()
