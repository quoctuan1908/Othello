import streamlit as st
import numpy as np
import time
# Giả định các module này đã có sẵn trong thư mục dự án
from othello.OthelloGame import OthelloGame
from othello.pytorch.NNet import NNetWrapper
from MCTS import MCTS
from utils import dotdict

# Khởi tạo game (6x6)
g = OthelloGame(6)

# Cấu hình MCTS
args = dotdict({
    'numMCTSSims': 50,  # Số lần mô phỏng MCTS
    'cpuct': 1.0,
})

# Tải mô hình AI
try:
    nnet = NNetWrapper(g)
    # Đảm bảo đường dẫn này đúng
    nnet.load_checkpoint('./model/', 'best.pth.tar') 
    mcts = MCTS(g, nnet, args)
    AI_LOADED = True
except Exception as e:
    st.error(f"⚠️ Lỗi khi tải mô hình AI hoặc thư viện: {e}")
    st.warning("Ứng dụng vẫn chạy nhưng AI sẽ không thể hoạt động.")
    AI_LOADED = False

if "board" not in st.session_state:
    st.session_state.board = g.getInitBoard()
    st.session_state.cur_player = 1 
    st.session_state.ai_turn = False # Cờ báo hiệu lượt AI
    st.session_state.game_over = False
    st.session_state.last_score_diff = 0

def play_move(x, y):
    if st.session_state.game_over or st.session_state.cur_player != 1:
        return
        
    n = g.n
    a = n * x + y
    valids = g.getValidMoves(st.session_state.board, st.session_state.cur_player)
    print(valids)
    # 1. Xử lý nước đi của Người chơi
    if valids[a]:
        st.session_state.board, st.session_state.cur_player = g.getNextState(
            st.session_state.board, st.session_state.cur_player, a
        )
        st.session_state.last_score_diff = g.getScore(st.session_state.board, 1)
        # 2. Kiểm tra kết thúc game sau nước đi của Người chơi
        if g.getGameEnded(st.session_state.board, st.session_state.cur_player) == 0:
            # Game chưa kết thúc -> Chuyển sang lượt AI
            st.session_state.ai_turn = True
        else:
            # Game kết thúc
            st.session_state.game_over = True

def reset_game():
    st.session_state.board = g.getInitBoard()
    st.session_state.cur_player = 1
    st.session_state.ai_turn = False
    st.session_state.game_over = False

st.title("Othello AI - Powered by AlphaZero")

board = st.session_state.board
n = g.n
game_over = st.session_state.game_over

# Chia layout: bàn cờ (trái) và stats (phải)
left_col, right_col = st.columns([3, 1])

# --- HIỂN THỊ BÀN CỜ ---
with left_col:
    st.subheader("Bàn cờ")
    
    # Lấy các nước đi hợp lệ cho người chơi hiện tại (để làm mờ ô không hợp lệ)
    valids = g.getValidMoves(board, st.session_state.cur_player)
    
    # Tạo container cho bàn cờ để kiểm soát layout tốt hơn
    board_container = st.container()
    
    with board_container:
        # Sử dụng các cột để mô phỏng bàn cờ
        for x in range(n):
            cols = st.columns(n)
            for y in range(n):
                a = n * x + y
                piece = board[x][y]
                
                # Xác định nhãn và style
                if piece == 1:
                    label = "⚪" 
                    style = "font-size: 24px;"
                elif piece == -1:
                    label = "⚫"
                    style = "font-size: 24px;"
                else:
                    label = " "
                    style = "font-size: 24px;"
                    # Thêm dấu chấm nhỏ nếu đó là nước đi hợp lệ của người chơi (1)
                    if st.session_state.cur_player == 1 and valids[a]:
                        label = "🔸"
                        style = "font-size: 16px; color: #aaa;" # Làm mờ dấu chấm

                # Tạo nút bấm
                if cols[y].button(label, key=f"{x}-{y}", help="Nhấn để đánh"):
                    if not game_over and st.session_state.cur_player == 1:
                        play_move(x, y)
                        st.rerun() # Kích hoạt render lại ngay lập tức sau nước người chơi

# --- HIỂN THỊ TRẠNG THÁI ---
with right_col:
    st.subheader("Trạng thái trận đấu")

    white_count = np.sum(board == 1)
    black_count = np.sum(board == -1)

    st.markdown(f"⚪ Quân bạn: `{white_count}`")
    st.markdown(f"⚫ Quân AI: `{black_count}`")

    st.divider()
    
    
    score_diff = st.session_state.last_score_diff
    if score_diff > 0:
        st.success(f"📈 Bạn đang dẫn: +{score_diff}")
    elif score_diff < 0:
        st.error(f"📉 AI đang dẫn: {score_diff}")
    else:
        st.info("⚖️ Điểm số đang Hòa")

    st.divider()
    
    if not game_over:
        if st.session_state.cur_player == 1:
            st.info("Đến lượt của (⚪)")
        elif st.session_state.ai_turn:
             st.warning("Đang chờ AI (⚫) đánh...")
        else:
             st.warning("Đến lượt AI (⚫)")
    
    # --- Trạng thái kết thúc ---
    result = g.getGameEnded(board, 1) # Kiểm tra kết quả từ góc nhìn của Người chơi (1)
    if result != 0 or game_over:
        st.divider()
        if result == 1:
            st.success("Bạn thắng!")
        elif result == -1:
            st.error("AI thắng!")
        else:
            st.info("Hòa!")
            
        st.session_state.game_over = True

        if st.button("🔄 Chơi lại", on_click=reset_game):
            # Hàm reset_game đã gọi st.session_state = ...
            st.rerun() 
    else:
        st.divider()
        st.caption("Ván đấu đang diễn ra...")

if st.session_state.ai_turn and AI_LOADED and not game_over:
    st.session_state.ai_turn = False 

    if g.getGameEnded(st.session_state.board, st.session_state.cur_player) == 0:
        
        time.sleep(0.2) 

        # Lượt AI
        with st.spinner('AI đang tính toán...'):
            canonical = g.getCanonicalForm(st.session_state.board, st.session_state.cur_player)
            ai_action = np.argmax(mcts.getActionProb(canonical, temp=0)) 
        
        # 2. Thực hiện nước đi của AI (có thể là một nước Pass)
        st.session_state.board, st.session_state.cur_player = g.getNextState(
            st.session_state.board, st.session_state.cur_player, ai_action
        )
        st.session_state.last_score_diff = g.getScore(st.session_state.board, 1)

        if g.getGameEnded(st.session_state.board, st.session_state.cur_player) != 0:
            st.session_state.game_over = True
            
        st.rerun()
        
    elif g.getGameEnded(st.session_state.board, st.session_state.cur_player) != 0:
        st.session_state.game_over = True
        st.rerun() # Bắt buộc phải rerun để cập nhật hiển thị kết quả
        
    else:
        
        st.session_state.game_over = True
        st.rerun()