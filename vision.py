import cv2
import numpy as np

ROWS, COLS = 6, 7

def _largest_rect_contour(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.08*h*w:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04*peri, True)
        if len(approx) == 4 and area > best_area:
            best, best_area = approx.reshape(4,2).astype(np.float32), area
    return best

def order_points(pts):
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)

def warp_board(img, corners, size=(700, 600)):
    corners = order_points(np.asarray(corners, dtype=np.float32))
    W,H = size
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, M, (W,H))

def classify_cells(board_bgr):
    # Robust-ish color classifier for common red/yellow Connect Four pieces.
    H,W = board_bgr.shape[:2]
    out = np.zeros((ROWS,COLS), dtype=np.int8)
    for r in range(ROWS):
        for c in range(COLS):
            x0, x1 = int(c*W/COLS), int((c+1)*W/COLS)
            y0, y1 = int(r*H/ROWS), int((r+1)*H/ROWS)
            cell = board_bgr[y0:y1, x0:x1]
            hh, ww = cell.shape[:2]
            # Sample center area, avoiding the plastic grid edges.
            padx, pady = int(ww*.22), int(hh*.22)
            roi = cell[pady:hh-pady, padx:ww-padx]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            Hh, S, V = [x.astype(np.float32) for x in cv2.split(hsv)]
            red = (((Hh < 10) | (Hh > 170)) & (S > 80) & (V > 70)).mean()
            yellow = ((Hh > 18) & (Hh < 45) & (S > 70) & (V > 80)).mean()
            if red > .10 and red > yellow:
                out[r,c] = 1
            elif yellow > .10:
                out[r,c] = 2
            else:
                out[r,c] = 0
    return out

def detect_board(img_bgr):
    pts = _largest_rect_contour(img_bgr)
    if pts is None:
        return None, None, "Automatic board detection failed. Use the manual-crop option."
    warped = warp_board(img_bgr, pts)
    board = classify_cells(warped)
    return board, warped, "Automatic detection"
