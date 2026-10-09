import streamlit as st
import cv2, numpy as np
from PIL import Image
from solver import best_move, valid_moves, winner, drop
from vision import detect_board, classify_cells, warp_board

st.set_page_config(page_title="Connect 4 Photo Solver", page_icon="🟡", layout="centered")
st.title("🟡 Connect 4 Photo Solver")
st.caption("Take a picture of the board → detect the position → calculate the strongest move.")

uploaded = st.file_uploader("Upload a photo of the Connect 4 board", type=["jpg","jpeg","png","webp"])

if uploaded:
    data = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Your photo", use_container_width=True)

    board, warped, msg = detect_board(img)
    st.write(msg)

    if board is not None:
        st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), caption="Detected board", use_container_width=True)
        st.write("Detected position:")
        st.dataframe(board, hide_index=True, use_container_width=True)

        me = st.selectbox("Which color are YOU?", ["Red", "Yellow"])
        me_num = 1 if me == "Red" else 2

        # Validate gravity: pieces should be stacked from bottom upward.
        gravity_ok = True
        for c in range(7):
            seen_empty = False
            for r in range(5, -1, -1):
                if board[r,c] == 0:
                    seen_empty = True
                elif seen_empty:
                    gravity_ok = False
        if not gravity_ok:
            st.warning("The photo detector may have misread a piece. Check the detected grid before trusting the answer.")

        w = winner(board)
        if w:
            st.error("This position already contains a winning four.")
        elif all(board[0,c] != 0 for c in range(7)):
            st.info("The board is full — draw.")
        else:
            limit = st.slider("Search time (seconds)", 1, 15, 5)
            if st.button("🚀 Find my best move", type="primary"):
                with st.spinner("Calculating..."):
                    col, reason, depth = best_move(board.tolist(), me_num, float(limit))
                if col is None:
                    st.info("No legal moves.")
                else:
                    st.success(f"PLAY COLUMN {col+1}  🎯")
                    st.write(f"**{reason}.**")
                    st.caption(f"Columns are numbered left → right. Search depth completed: {depth}.")
                    st.balloons()

st.divider()
st.markdown("""
### How to get the best results
- Photograph the entire board straight-on.
- Make sure the holes/pieces are visible and not covered by glare.
- Keep the board roughly centered.
- If the detected grid is wrong, retake the photo with better lighting.

**Important:** The solver gives the mathematically strongest move it can calculate within the selected time. A time-limited search is not a proof that every position is a forced win.
""")
