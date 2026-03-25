"""
╔══════════════════════════════════════════════════════════════╗
║   HAND GESTURE DATA COLLECTOR                                ║
║   Builds a labelled landmark database for training           ║
╠══════════════════════════════════════════════════════════════╣
║  INSTALL:                                                    ║
║    pip install opencv-python mediapipe                       ║
║                                                              ║
║  RUN:                                                        ║
║    python collect_data.py                                    ║
╠══════════════════════════════════════════════════════════════╣
║  CONTROLS (inside the OpenCV window):                        ║
║   1  →  Label: MOVE      (index finger up)                   ║
║   2  →  Label: PINCH     (thumb + index)                     ║
║   3  →  Label: PEACE     (index + middle up)                 ║
║   4  →  Label: FIST      (all fingers closed)                ║
║   5  →  Label: NONE      (no hand / rest)                    ║
║   SPACE  →  Pause / Resume auto-capture                      ║
║   R  →  Reset counter for current label                      ║
║   S  →  Save & show stats                                    ║
║   Q  →  Quit and save                                        ║
╠══════════════════════════════════════════════════════════════╣
║  OUTPUT:  hand_data/landmarks.csv                            ║
║           Columns: label, lm0_x, lm0_y, lm0_z, … lm20_z    ║
║           (63 landmark values, normalised to wrist origin)   ║
╚══════════════════════════════════════════════════════════════╝
"""

import cv2
import math
import time
import sys
import os
import csv
import numpy as np
import mediapipe as mp

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "camera_index":  1,
    "camera_width":  1280,
    "camera_height": 720,
    "flip_camera":   True,
    "output_dir":    "hand_data",
    "output_file":   "landmarks.csv",
    # How many frames per second to auto-capture while key held
    "capture_fps":   12,
    # Target samples per class (just for progress display)
    "target_per_class": 1000,
}

LABELS = {
    ord('1'): "MOVE",
    ord('2'): "PINCH",
    ord('3'): "PEACE",
    ord('4'): "FIST",
    ord('5'): "NONE",
}

LABEL_COLORS = {
    "MOVE":  (0, 229, 255),
    "PINCH": (0, 255, 136),
    "PEACE": (123, 47, 255),
    "FIST":  (0, 100, 255),
    "NONE":  (160, 160, 160),
}

# CSV header
HEADER = ["label"] + [f"lm{i}_{ax}" for i in range(21) for ax in ("x","y","z")]

# ─────────────────────────────────────────────
#  mediapipe version detection (same as main)
# ─────────────────────────────────────────────
def get_mp_version():
    try:
        import importlib.metadata
        v = importlib.metadata.version("mediapipe").split(".")
        return int(v[0]), int(v[1])
    except Exception:
        return 0, 9

MP_MAJOR, MP_MINOR = get_mp_version()
USE_NEW_API = (MP_MAJOR > 0) or (MP_MAJOR == 0 and MP_MINOR >= 10)

# ─────────────────────────────────────────────
#  NORMALISATION
# ─────────────────────────────────────────────
def normalise_landmarks(lm_list):
    """
    Translate to wrist-origin, scale by hand size (wrist→middle-MCP),
    then flatten to a 63-element list [x0,y0,z0, x1,y1,z1, …].

    This makes the feature vector:
      • Translation-invariant  (position in frame doesn't matter)
      • Scale-invariant        (distance from camera doesn't matter)
      • Suitable for a classifier that learns shape, not position
    """
    xs = [l[0] for l in lm_list]
    ys = [l[1] for l in lm_list]
    zs = [l[2] for l in lm_list]

    # Wrist is landmark 0
    wx, wy, wz = xs[0], ys[0], zs[0]

    # Translate to wrist origin
    xs = [x - wx for x in xs]
    ys = [y - wy for y in ys]
    zs = [z - wz for z in zs]

    # Scale: distance from wrist (0) to middle MCP (9)
    scale = math.hypot(xs[9], ys[9]) + 1e-9
    xs = [x / scale for x in xs]
    ys = [y / scale for y in ys]
    zs = [z / scale for z in zs]

    flat = []
    for i in range(21):
        flat.extend([xs[i], ys[i], zs[i]])
    return flat

# ─────────────────────────────────────────────
#  AUGMENTATION  (applied at capture time)
# ─────────────────────────────────────────────
def augment(flat_63, n=4):
    """
    Returns the original + n augmented copies.
    Augmentations: small rotation, Gaussian jitter, horizontal flip (mirroring).
    All done in 2-D (x, y) only — z is jitter only.
    """
    samples = [flat_63]
    arr = np.array(flat_63).reshape(21, 3)

    for _ in range(n):
        a = arr.copy()

        # 1. Small random rotation (±15°)
        angle = np.random.uniform(-15, 15) * math.pi / 180
        ca, sa = math.cos(angle), math.sin(angle)
        rot = np.array([[ca, -sa], [sa, ca]])
        a[:, :2] = (rot @ a[:, :2].T).T

        # 2. Gaussian coordinate jitter (σ=0.02 in normalised units)
        a += np.random.normal(0, 0.02, a.shape)

        # 3. 50 % chance: flip hand horizontally (mirror → simulate other hand)
        if np.random.rand() < 0.5:
            a[:, 0] = -a[:, 0]

        # 4. Small scale perturbation (±10 %)
        scale_p = np.random.uniform(0.90, 1.10)
        a[:, :2] *= scale_p

        samples.append(a.flatten().tolist())

    return samples

# ─────────────────────────────────────────────
#  DRAW HELPERS
# ─────────────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),(19,20),
]

def draw_hand(frame, lm, color, w, h):
    for a, b in CONNECTIONS:
        ax, ay = int(lm[a][0] * w), int(lm[a][1] * h)
        bx, by = int(lm[b][0] * w), int(lm[b][1] * h)
        cv2.line(frame, (ax, ay), (bx, by), color, 1, cv2.LINE_AA)
    for i, l in enumerate(lm):
        px, py = int(l[0] * w), int(l[1] * h)
        cv2.circle(frame, (px, py), 5 if i in (4,8) else 3, color, -1)

def draw_text(frame, text, pos, color=(0, 229, 200), scale=0.55, thickness=1):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0,0,0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)

def draw_panel(frame, lines, x, y, color=(0, 229, 255)):
    line_h, pad = 22, 10
    pw = max(len(l) for l in lines) * 9 + pad * 2
    ph = len(lines) * line_h + pad * 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x+pw, y+ph), (10,14,20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (x, y), (x+pw, y+ph), color, 1)
    for i, line in enumerate(lines):
        draw_text(frame, line, (x+pad, y+pad+14+i*line_h), color)

def progress_bar(frame, x, y, w, val, maxi, color):
    cv2.rectangle(frame, (x, y), (x+w, y+16), (40,40,40), -1)
    fill = int(min(1.0, val/max(maxi,1)) * w)
    cv2.rectangle(frame, (x, y), (x+fill, y+16), color, -1)
    cv2.rectangle(frame, (x, y), (x+w, y+16), color, 1)
    pct = int(100 * val / max(maxi, 1))
    draw_text(frame, f"{val}/{maxi} ({pct}%)", (x+4, y+13), (255,255,255), 0.38)

# ─────────────────────────────────────────────
#  MAIN COLLECTOR
# ─────────────────────────────────────────────
def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    csv_path = os.path.join(CONFIG["output_dir"], CONFIG["output_file"])

    # Load existing data to continue from where we left off
    counts = {name: 0 for name in LABELS.values()}
    file_exists = os.path.isfile(csv_path)

    if file_exists:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lbl = row.get("label", "")
                if lbl in counts:
                    counts[lbl] += 1
        print(f"\n  Resuming — existing data: {counts}")
    else:
        print("\n  Starting fresh dataset.")

    csv_file = open(csv_path, "a", newline="")
    writer   = csv.writer(csv_file)
    if not file_exists:
        writer.writerow(HEADER)

    # ── Init mediapipe ──
    if USE_NEW_API:
        import urllib.request
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
            num_hands=1,
            min_hand_detection_confidence=0.70,
            min_hand_presence_confidence=0.70,
            min_tracking_confidence=0.60,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        landmarker = mp_vision.HandLandmarker.create_from_options(opts)
        legacy     = None
    else:
        landmarker = None
        legacy     = mp.solutions.hands.Hands(
            max_num_hands=1, model_complexity=1,
            min_detection_confidence=0.75, min_tracking_confidence=0.65)

    cap = cv2.VideoCapture(CONFIG["camera_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CONFIG["camera_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera_height"])
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        sys.exit(1)

    active_label   = "MOVE"
    paused         = False
    last_capture_t = 0
    frame_count    = 0
    fps_time       = time.time()
    fps            = 0
    flash_msg      = ""
    flash_t        = 0
    total_saved    = sum(counts.values())
    capture_interval = 1.0 / CONFIG["capture_fps"]

    print("\n  Controls: 1=MOVE  2=PINCH  3=PEACE  4=FIST  5=NONE")
    print("            SPACE=pause  R=reset  S=stats  Q=quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if CONFIG["flip_camera"]:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        frame_count += 1
        now = time.time()

        if now - fps_time >= 1.0:
            fps = frame_count; frame_count = 0; fps_time = now

        # ── Run detection ──
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lm_raw = None

        if USE_NEW_API:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_img)
            if result.hand_landmarks:
                lm_raw = [(l.x, l.y, l.z) for l in result.hand_landmarks[0]]
        else:
            rgb.flags.writeable = False
            res = legacy.process(rgb)
            rgb.flags.writeable = True
            if res.multi_hand_landmarks:
                lm_raw = [(l.x, l.y, l.z) for l in res.multi_hand_landmarks[0].landmark]

        label_color = LABEL_COLORS.get(active_label, (200, 200, 200))

        # ── Auto-capture ──
        captured_now = False
        if lm_raw and not paused and (now - last_capture_t) >= capture_interval:
            flat = normalise_landmarks(lm_raw)
            samples = augment(flat, n=3)   # 1 real + 3 augmented per frame
            for s in samples:
                writer.writerow([active_label] + s)
            counts[active_label] += len(samples)
            total_saved           += len(samples)
            last_capture_t         = now
            captured_now           = True
            csv_file.flush()

        # ── Draw skeleton ──
        if lm_raw:
            col = (0, 255, 100) if captured_now else label_color
            draw_hand(frame, lm_raw, col, w, h)

        # ── HUD panels ──
        status_lines = [
            f"FPS      : {fps}",
            f"Label    : {active_label}",
            f"Total    : {total_saved}",
            f"Status   : {'PAUSED' if paused else 'CAPTURING'}",
        ]
        draw_panel(frame, status_lines, 10, 10, label_color)

        # Per-label progress bars
        bar_x, bar_y = 10, 130
        target = CONFIG["target_per_class"]
        for name, count in counts.items():
            col = LABEL_COLORS[name]
            draw_text(frame, f"{name}:", (bar_x, bar_y + 12), col, 0.42)
            progress_bar(frame, bar_x + 75, bar_y, 220, count, target, col)
            bar_y += 24

        # Key guide
        key_guide = ["1:MOVE 2:PINCH 3:PEACE 4:FIST 5:NONE",
                     "SPACE:pause  R:reset  S:stats  Q:quit"]
        draw_panel(frame, key_guide, 10, h - 65, (80, 90, 100))

        # Flash message
        if flash_msg and (now - flash_t < 1.5):
            draw_text(frame, flash_msg, (w//2 - 150, h//2),
                      (0, 255, 200), 0.9, 2)

        # Capture flash indicator
        if captured_now:
            cv2.circle(frame, (w - 24, 24), 8, (0, 255, 100), -1)

        # Paused overlay
        if paused:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (w,h), (20,20,80), -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            draw_text(frame, "PAUSED  —  press SPACE to resume",
                      (w//2 - 200, h//2), (0,229,255), 0.8, 2)

        cv2.imshow("Hand Gesture Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF

        if key in LABELS:
            active_label = LABELS[key]
            flash_msg = f"Label → {active_label}"; flash_t = now
            print(f"  Label changed to: {active_label}  (count: {counts[active_label]})")

        elif key == ord(' '):
            paused = not paused
            print(f"  {'Paused' if paused else 'Resumed'}")

        elif key == ord('r'):
            print(f"  Resetting counter for {active_label} "
                  f"(was {counts[active_label]}, data still saved)")
            counts[active_label] = 0
            flash_msg = f"Counter reset for {active_label}"; flash_t = now

        elif key == ord('s'):
            print(f"\n  ── Dataset stats ──")
            for name, cnt in counts.items():
                bar = "█" * min(40, cnt // 25)
                print(f"    {name:<8} {cnt:>6}  {bar}")
            print(f"    TOTAL   {total_saved:>6}")
            print(f"    File    {csv_path}\n")
            flash_msg = f"Stats printed to console"; flash_t = now

        elif key == ord('q'):
            print("\n  Exiting and saving …")
            break

    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()
    if USE_NEW_API and landmarker:
        landmarker.close()
    if legacy:
        legacy.close()

    print("\n  ── Final counts ──")
    for name, cnt in counts.items():
        print(f"    {name:<8} {cnt:>6} samples")
    print(f"  Saved to: {csv_path}")
    print(f"  Next step: python train_model.py\n")


if __name__ == "__main__":
    main()
