"""
╔══════════════════════════════════════════════════════════════╗
║    VISION CURSOR CONTROLLER  —  ML-Powered Edition           ║
║    Drop-in replacement using your trained gesture model      ║
╠══════════════════════════════════════════════════════════════╣
║  INSTALL:                                                    ║
║    pip install opencv-python mediapipe pyautogui             ║
║    pip install scikit-learn joblib numpy                     ║
║                                                              ║
║  WORKFLOW:                                                   ║
║    1. python collect_data.py   → build hand_data/            ║
║    2. python train_model.py    → train gesture_model.pkl     ║
║    3. python cursor_controller.py  ← this file               ║
╠══════════════════════════════════════════════════════════════╣
║  FALLBACK:  if model file not found, reverts to              ║
║             rule-based detection automatically               ║
╠══════════════════════════════════════════════════════════════╣
║  GESTURES:                                                   ║
║   ☝  Index finger up   →  Move cursor                       ║
║   Pinch (thumb+index)  →  Left click                         ║
║   Peace / V sign       →  Right click                        ║
║   Fist                 →  Click & drag                       ║
╠══════════════════════════════════════════════════════════════╣
║  CONTROLS:                                                   ║
║   Q  →  Quit                                                 ║
║   +  →  Increase sensitivity                                 ║
║   -  →  Decrease sensitivity                                 ║
║   M  →  Toggle ML / rule-based detector                      ║
╚══════════════════════════════════════════════════════════════╝
"""

import cv2
import pyautogui
import math
import time
import sys
import os
import urllib.request
import numpy as np

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "camera_index":     1,
    "sensitivity":      1.6,
    "frame_reduction":  80,
    "flip_camera":      True,
    "camera_width":     1280,
    "camera_height":    720,

    # Rule-based fallback thresholds
    "pinch_enter":      0.28,
    "pinch_exit":       0.38,

    "click_cooldown":   0.35,
    "confirm_frames":   2,

    "kalman_Q":         0.01,
    "kalman_R":         0.04,
    "dead_zone_px":     2.5,
    "alpha_fast":       0.80,
    "alpha_slow":       0.50,
    "fast_speed":       80.0,
    "slow_speed":       8.0,

    # ML model
    "model_path":       "hand_data/gesture_model.pkl",
    # Minimum confidence for ML prediction (0–1). Below this → fallback to rule-based.
    "ml_confidence":    0.70,
}

MODEL_LABELS = ["MOVE", "PINCH", "PEACE", "FIST", "NONE"]

pyautogui.FAILSAFE = True
pyautogui.PAUSE    = 0
SCREEN_W, SCREEN_H = pyautogui.size()

# ─────────────────────────────────────────────
#  mediapipe version detection
# ─────────────────────────────────────────────
import mediapipe as mp

def get_mp_version():
    try:
        import importlib.metadata
        v = importlib.metadata.version("mediapipe").split(".")
        return int(v[0]), int(v[1])
    except Exception:
        return 0, 9

MP_MAJOR, MP_MINOR = get_mp_version()
USE_NEW_API = (MP_MAJOR > 0) or (MP_MAJOR == 0 and MP_MINOR >= 10)
print(f"\n  mediapipe {MP_MAJOR}.{MP_MINOR}  →  "
      f"{'Tasks API' if USE_NEW_API else 'Solutions API'}")

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def draw_text(frame, text, pos, color=(0, 255, 200), scale=0.55, thickness=1):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)

def draw_panel(frame, lines, x, y, color=(0, 229, 255)):
    line_h = 22; pad = 10
    pw = max(len(l) for l in lines) * 9 + pad * 2
    ph = len(lines) * line_h + pad * 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + pw, y + ph), (10, 14, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (x, y), (x + pw, y + ph), color, 1)
    for i, line in enumerate(lines):
        draw_text(frame, line, (x + pad, y + pad + 14 + i * line_h), color)

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),(19,20),
]

def draw_hand(frame, lm, gesture, w, h):
    col = (0,255,136) if gesture["pinch"] else (123,47,255) if gesture["peace"] else (0,229,255)
    for a, b in CONNECTIONS:
        ax, ay = int(lm[a].x * w), int(lm[a].y * h)
        bx, by = int(lm[b].x * w), int(lm[b].y * h)
        cv2.line(frame, (ax, ay), (bx, by), col, 1, cv2.LINE_AA)
    for i, l in enumerate(lm):
        px, py = int(l.x * w), int(l.y * h)
        r = 5 if i in (4, 8) else 3
        cv2.circle(frame, (px, py), r,
                   col if i in (4, 8) else (200, 220, 255), -1)

# ─────────────────────────────────────────────
#  LANDMARK WRAPPER
# ─────────────────────────────────────────────
class LM:
    __slots__ = ("x", "y", "z")
    def __init__(self, x, y, z=0.0):
        self.x = x; self.y = y; self.z = z

def unpack_legacy(results):
    if not results.multi_hand_landmarks:
        return None
    return [LM(l.x, l.y, l.z) for l in results.multi_hand_landmarks[0].landmark]

def unpack_new(result):
    if not result.hand_landmarks:
        return None
    return [LM(l.x, l.y, l.z) for l in result.hand_landmarks[0]]

# ─────────────────────────────────────────────
#  KALMAN FILTER
# ─────────────────────────────────────────────
class Kalman1D:
    def __init__(self, init_val, Q, R):
        self.x = np.array([init_val, 0.0])
        self.P = np.eye(2)
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = np.diag([Q, Q * 0.1])
        self.R = np.array([[R]])

    def update(self, z):
        x_p = self.F @ self.x
        P_p = self.F @ self.P @ self.F.T + self.Q
        S   = self.H @ P_p @ self.H.T + self.R
        K   = P_p @ self.H.T / S[0, 0]
        y   = z - (self.H @ x_p)[0]
        self.x = x_p + K.flatten() * y
        self.P = (np.eye(2) - np.outer(K, self.H)) @ P_p
        return self.x[0], self.x[1]

class CursorSmoother:
    def __init__(self):
        self.kx = Kalman1D(SCREEN_W/2, CONFIG["kalman_Q"], CONFIG["kalman_R"])
        self.ky = Kalman1D(SCREEN_H/2, CONFIG["kalman_Q"], CONFIG["kalman_R"])
        self.ox = SCREEN_W / 2; self.oy = SCREEN_H / 2

    def update(self, raw_x, raw_y):
        kx, vx = self.kx.update(raw_x)
        ky, vy = self.ky.update(raw_y)
        speed  = math.hypot(vx, vy)
        if speed < CONFIG["dead_zone_px"]:
            kx, ky = self.ox, self.oy
        hi = CONFIG["fast_speed"]; lo = CONFIG["slow_speed"]
        af = CONFIG["alpha_fast"]; as_ = CONFIG["alpha_slow"]
        t     = max(0.0, min(1.0, (speed - lo) / max(hi - lo, 1.0)))
        alpha = as_ + (af - as_) * t
        self.ox = self.ox + alpha * (kx - self.ox)
        self.oy = self.oy + alpha * (ky - self.oy)
        return int(self.ox), int(self.oy)

# ─────────────────────────────────────────────
#  FEATURE HELPERS  (must match train_model.py)
# ─────────────────────────────────────────────
def normalise_landmarks(lm_list):
    xs = [l.x for l in lm_list]; ys = [l.y for l in lm_list]; zs = [l.z for l in lm_list]
    wx, wy, wz = xs[0], ys[0], zs[0]
    xs = [x - wx for x in xs]; ys = [y - wy for y in ys]; zs = [z - wz for z in zs]
    scale = math.hypot(xs[9], ys[9]) + 1e-9
    xs = [x / scale for x in xs]; ys = [y / scale for y in ys]; zs = [z / scale for z in zs]
    flat = []
    for i in range(21):
        flat.extend([xs[i], ys[i], zs[i]])
    return flat

def add_engineered_features(X_raw):
    """Mirror of train_model.py — must stay in sync."""
    n = X_raw.shape[0]
    feat_extra = np.zeros((n, 16), dtype=np.float32)

    def lm(arr, idx):
        base = idx * 3
        return arr[:, base:base+3]

    dist_pairs = [
        (4,8),(4,12),(4,16),(4,20),(8,12),
        (0,9),(5,17),(8,0),(12,0),(20,0),
    ]
    hand_scale = np.linalg.norm(lm(X_raw,9) - lm(X_raw,0), axis=1, keepdims=True) + 1e-9
    for col_i, (a, b) in enumerate(dist_pairs):
        d = np.linalg.norm(lm(X_raw,a) - lm(X_raw,b), axis=1)
        feat_extra[:, col_i] = d / hand_scale[:, 0]

    finger_joints = [(5,6,7,8),(9,10,11,12),(13,14,15,16),(17,18,19,20),(1,2,3,4)]
    for col_i, (mcp_i, pip_i, _, tip_i) in enumerate(finger_joints):
        mcp = lm(X_raw, mcp_i); pip = lm(X_raw, pip_i); tip = lm(X_raw, tip_i)
        u = pip - mcp; v = tip - pip
        cos_a = np.einsum("ij,ij->i", u, v) / (
            np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
        feat_extra[:, 10 + col_i] = np.arccos(np.clip(cos_a, -1.0, 1.0))

    t_vec = lm(X_raw,4)[:,:2] - lm(X_raw,2)[:,:2]
    i_vec = lm(X_raw,8)[:,:2] - lm(X_raw,5)[:,:2]
    feat_extra[:, 15] = (t_vec[:,0]*i_vec[:,1] - t_vec[:,1]*i_vec[:,0]) / (hand_scale[:,0]**2 + 1e-9)

    return np.concatenate([X_raw.astype(np.float32), feat_extra], axis=1)

def build_feature_vector(lm):
    flat = normalise_landmarks(lm)
    X    = np.array(flat, dtype=np.float32).reshape(1, -1)
    return add_engineered_features(X)

# ─────────────────────────────────────────────
#  ML GESTURE PREDICTOR
# ─────────────────────────────────────────────
class MLGesturePredictor:
    def __init__(self, model_path):
        try:
            import joblib
            bundle       = joblib.load(model_path)
            self.model   = bundle["model"]
            self.inv_map = bundle["inv_map"]
            self.labels  = bundle["labels"]
            self.loaded  = True
            print(f"  ✓  ML model loaded: {model_path}  "
                  f"(test acc {bundle.get('test_acc',0):.4f})")
        except Exception as e:
            print(f"  ✗  Could not load ML model ({e}) — using rule-based fallback.")
            self.loaded  = False

    def predict(self, lm_list):
        """Returns (label_str, confidence) or (None, 0) if not loaded."""
        if not self.loaded:
            return None, 0.0
        try:
            X    = build_feature_vector(lm_list)
            prob = self.model.predict_proba(X)[0]
            idx  = int(np.argmax(prob))
            conf = float(prob[idx])
            label = self.inv_map.get(idx, "NONE")
            return label, conf
        except Exception:
            return None, 0.0

# ─────────────────────────────────────────────
#  RULE-BASED GESTURE DETECTOR  (original, fallback)
# ─────────────────────────────────────────────
def _vec3(a, b):
    return np.array([b.x-a.x, b.y-a.y, b.z-a.z])

def _angle(u, v):
    c = np.dot(u,v) / (np.linalg.norm(u)*np.linalg.norm(v) + 1e-9)
    return math.degrees(math.acos(max(-1.0,min(1.0,c))))

def finger_extended(tip_idx, pip_idx, mcp_idx, lm):
    try:
        return _angle(_vec3(lm[mcp_idx],lm[pip_idx]),
                      _vec3(lm[pip_idx],lm[tip_idx])) > 140.0
    except Exception:
        return lm[tip_idx].y < lm[mcp_idx].y

def hand_scale(lm):
    return dist(lm[0], lm[9]) + 1e-9

class RuleGestureDetector:
    def __init__(self):
        self.pinch = False; self.peace = False; self.fist = False
        self._cand = {"pinch":False,"peace":False,"fist":False}
        self._cnt  = {"pinch":0,    "peace":0,    "fist":0}

    def _confirm(self, key, candidate):
        conf = CONFIG["confirm_frames"]
        if candidate != self._cand[key]:
            self._cand[key] = candidate; self._cnt[key] = 1
        else:
            self._cnt[key] = min(self._cnt[key]+1, conf)
        return candidate if self._cnt[key] >= conf else getattr(self, key)

    def detect(self, lm):
        scale  = hand_scale(lm)
        pe, px = CONFIG["pinch_enter"], CONFIG["pinch_exit"]
        pd_n   = dist(lm[4], lm[8]) / scale
        raw_p  = pd_n < (px if self.pinch else pe)
        idx_up = finger_extended(8,7,5,lm); mid_up = finger_extended(12,11,9,lm)
        rng_up = finger_extended(16,15,13,lm); pky_up = finger_extended(20,19,17,lm)
        raw_peace = idx_up and mid_up and not rng_up and not pky_up and not raw_p
        raw_fist  = not idx_up and not mid_up and not rng_up and not pky_up
        self.pinch = self._confirm("pinch", raw_p)
        self.peace = self._confirm("peace", raw_peace)
        self.fist  = self._confirm("fist",  raw_fist)
        move = idx_up and not self.pinch and not self.peace and not self.fist
        return {
            "pinch": self.pinch, "peace": self.peace,
            "fist":  self.fist,  "move":  move, "pd_norm": pd_n,
        }

# ─────────────────────────────────────────────
#  UNIFIED GESTURE CONTROLLER
# ─────────────────────────────────────────────
class GestureController:
    """
    Wraps both ML and rule-based detectors.
    Falls back to rules when ML confidence is too low.
    Adds hysteresis + confirmation on top of ML predictions too.
    """
    def __init__(self, ml_predictor, rule_detector):
        self.ml   = ml_predictor
        self.rule = rule_detector
        self.use_ml = ml_predictor.loaded

        # Confirmation buffer for ML path
        self._ml_cand  = "NONE"
        self._ml_cnt   = 0

        # Action state
        self.is_dragging = False
        self.last_click  = 0

        # Stats
        self.ml_hits   = 0
        self.rule_hits = 0

    def _confirm_ml(self, raw_label):
        n = CONFIG["confirm_frames"]
        if raw_label != self._ml_cand:
            self._ml_cand = raw_label; self._ml_cnt = 1
        else:
            self._ml_cnt = min(self._ml_cnt + 1, n)
        if self._ml_cnt >= n:
            return raw_label
        # Hold previous (keep the last confirmed label)
        return getattr(self, "_confirmed_ml_label", "NONE")

    def detect(self, lm):
        """Returns unified gesture dict + metadata."""
        source = "rule"
        confidence = 1.0

        if self.use_ml:
            label, conf = self.ml.predict(lm)
            if label and conf >= CONFIG["ml_confidence"]:
                confirmed = self._confirm_ml(label)
                self._confirmed_ml_label = confirmed
                self.ml_hits += 1
                source      = "ml"
                confidence  = conf
                # Convert label string → gesture dict
                gesture = {
                    "pinch":   confirmed == "PINCH",
                    "peace":   confirmed == "PEACE",
                    "fist":    confirmed == "FIST",
                    "move":    confirmed == "MOVE",
                    "pd_norm": dist(lm[4], lm[8]) / (hand_scale(lm)),
                    "label":   confirmed,
                    "conf":    conf,
                    "source":  source,
                }
                return gesture

        # Rule-based fallback
        self.rule_hits += 1
        g = self.rule.detect(lm)
        if g["pinch"]:   label = "PINCH"
        elif g["peace"]: label = "PEACE"
        elif g["fist"]:  label = "FIST"
        elif g["move"]:  label = "MOVE"
        else:            label = "NONE"
        g["label"]  = label
        g["conf"]   = confidence
        g["source"] = source
        return g

    def act(self, gesture, cx, cy):
        now  = time.time()
        cool = CONFIG["click_cooldown"]
        action = None

        if gesture["pinch"]:
            if not self.is_dragging and (now - self.last_click) > cool:
                pyautogui.click(cx, cy)
                self.last_click = now
                action = "LEFT CLICK"
        elif gesture["peace"]:
            if (now - self.last_click) > cool:
                pyautogui.rightClick(cx, cy)
                self.last_click = now
                action = "RIGHT CLICK"
        elif gesture["fist"]:
            if not self.is_dragging:
                pyautogui.mouseDown(cx, cy)
                self.is_dragging = True
                action = "DRAG START"
        else:
            if self.is_dragging:
                pyautogui.mouseUp()
                self.is_dragging = False
                action = "DRAG END"

        return action

# ─────────────────────────────────────────────
#  INIT MEDIAPIPE
# ─────────────────────────────────────────────
def init_new_api():
    model_path = "hand_landmarker.task"
    if not os.path.exists(model_path):
        print("  Downloading hand_landmarker.task …")
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        urllib.request.urlretrieve(url, model_path)
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        num_hands=1, min_hand_detection_confidence=0.75,
        min_hand_presence_confidence=0.75, min_tracking_confidence=0.70,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)

def init_legacy_api():
    return mp.solutions.hands.Hands(
        max_num_hands=1, model_complexity=1,
        min_detection_confidence=0.80, min_tracking_confidence=0.70)

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  VISION CURSOR CONTROLLER  —  ML-Powered Edition")
    print("="*60)
    print(f"  Screen : {SCREEN_W} × {SCREEN_H}")

    ml_predictor  = MLGesturePredictor(CONFIG["model_path"])
    rule_detector = RuleGestureDetector()
    controller    = GestureController(ml_predictor, rule_detector)

    if USE_NEW_API:
        landmarker = init_new_api(); legacy_hands = None
    else:
        landmarker = None; legacy_hands = init_legacy_api()

    cap = cv2.VideoCapture(CONFIG["camera_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CONFIG["camera_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera_height"])
    if not cap.isOpened():
        print("ERROR: Cannot open camera."); sys.exit(1)

    smoother     = CursorSmoother()
    fps_time     = time.time(); frame_count = 0; fps = 0
    last_action  = ""; last_act_t = 0; click_total = 0
    cx_s = SCREEN_W // 2; cy_s = SCREEN_H // 2

    print("  M  →  toggle ML / rule-based")
    print("  Q  →  quit   +/- → sensitivity")
    print("="*60 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        if CONFIG["flip_camera"]:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        fr   = CONFIG["frame_reduction"]
        frame_count += 1
        if time.time() - fps_time >= 1.0:
            fps = frame_count; frame_count = 0; fps_time = time.time()

        cv2.rectangle(frame, (fr, fr), (w-fr, h-fr), (0,229,255), 1)
        draw_text(frame, "Detection Zone", (fr+6, fr-8), (0,229,255), 0.4)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lm  = None

        if USE_NEW_API:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_img)
            lm     = unpack_new(result)
        else:
            rgb.flags.writeable = False
            results = legacy_hands.process(rgb)
            rgb.flags.writeable = True
            lm = unpack_legacy(results)

        gesture_label = "---"

        if lm:
            gesture = controller.detect(lm)

            # Map index finger tip to screen
            zone_w = w - 2*fr; zone_h = h - 2*fr
            nx = max(0.0, min(1.0, (lm[8].x*w - fr) / zone_w))
            ny = max(0.0, min(1.0, (lm[8].y*h - fr) / zone_h))
            nx = max(0.0, min(1.0, 0.5 + (nx-0.5)*CONFIG["sensitivity"]))
            ny = max(0.0, min(1.0, 0.5 + (ny-0.5)*CONFIG["sensitivity"]))

            cx_s, cy_s = smoother.update(nx*SCREEN_W, ny*SCREEN_H)
            pyautogui.moveTo(cx_s, cy_s)

            src   = gesture["source"].upper()
            conf  = gesture.get("conf", 1.0)
            label = gesture.get("label", "NONE")
            gesture_label = f"{label}  [{src} {conf:.2f}]"

            action = controller.act(gesture, cx_s, cy_s)
            if action:
                last_action = action; last_act_t = time.time()
                if "CLICK" in action: click_total += 1

            draw_hand(frame, lm, gesture, w, h)

        # ── HUD ──
        total_pred = max(1, controller.ml_hits + controller.rule_hits)
        mode_str   = ("ML" if controller.use_ml else "RULES") + \
                     (f" ({100*controller.ml_hits//total_pred}% ML)" if controller.use_ml else "")

        draw_panel(frame, [
            f"FPS      : {fps}",
            f"Mode     : {mode_str}",
            f"Gesture  : {gesture_label}",
            f"Cursor   : {cx_s}, {cy_s}",
            f"Clicks   : {click_total}",
            f"Sensitiv : {CONFIG['sensitivity']:.1f}",
        ], 10, 10)

        draw_text(frame, f"{fps} FPS", (w-80, 25), (0,229,255), 0.55)

        if time.time() - last_act_t < 0.8 and last_action:
            col = ((0,255,136) if "LEFT" in last_action else
                   (123,47,255) if "RIGHT" in last_action else (0,229,255))
            draw_text(frame, last_action, (w//2-80, h-30), col, 0.75, 2)

        # Corner brackets
        for bx, by in [(0,0),(w,0),(0,h),(w,h)]:
            dx = 1 if bx==0 else -1; dy = 1 if by==0 else -1
            cv2.line(frame,(bx,by),(bx+dx*24,by),(0,229,255),2)
            cv2.line(frame,(bx,by),(bx,by+dy*24),(0,229,255),2)

        draw_text(frame, "Q:quit  M:toggle ML/rules  +/-:sensitivity",
                  (10, h-12), (80,90,100), 0.38)

        cv2.imshow("Vision Cursor Controller  —  ML Edition", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('m'):
            controller.use_ml = not controller.use_ml and ml_predictor.loaded
            print(f"  Detector: {'ML' if controller.use_ml else 'Rule-based'}")
        elif key in (ord('+'), ord('=')):
            CONFIG["sensitivity"] = min(3.0, round(CONFIG["sensitivity"]+0.1,1))
            print(f"  Sensitivity → {CONFIG['sensitivity']}")
        elif key == ord('-'):
            CONFIG["sensitivity"] = max(0.5, round(CONFIG["sensitivity"]-0.1,1))
            print(f"  Sensitivity → {CONFIG['sensitivity']}")

    cap.release()
    cv2.destroyAllWindows()
    if USE_NEW_API and landmarker: landmarker.close()
    if legacy_hands: legacy_hands.close()
    print(f"\n  Session: {click_total} clicks | "
          f"ML:{controller.ml_hits} Rule:{controller.rule_hits}")
    print("Stopped.")


if __name__ == "__main__":
    main()
