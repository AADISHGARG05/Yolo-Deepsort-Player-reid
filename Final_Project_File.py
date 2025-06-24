# === PLAYER RE-IDENTIFICATION SYSTEM ===
# Author: Aadish Garg
# Internship Project Submission
# Objective: Track and count unique players in a 15-second video using YOLOv11 and DeepSORT.

import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict, deque
import numpy as np

# === CONFIGURATION SECTION ===
MODEL_PATH = "best.pt"                         # Trained YOLOv11 model (fine-tuned for player detection)(Given in the File)
VIDEO_PATH = "15sec_input_720p.mp4"            # Input video file
OUTPUT_PATH = "output_final.mp4"               # Output file with tracking annotations

CONFIDENCE_THRESHOLD = 0.75                    # Minimum confidence to consider detection valid
MIN_BOX_AREA = 800                             # Ignore small bounding boxes (likely noise)
TRACK_CLASSES = [2]                            # Class ID for 'player' in YOLO's output
REID_FRAME_TOLERANCE = 10                      # Max frames to tolerate before considering a track lost
IOU_THRESHOLD = 0.4                            # IOU threshold for Re-ID decision

# === VIDEO SETUP ===
cap = cv2.VideoCapture(VIDEO_PATH)
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# === LOAD YOLO MODEL AND INITIALIZE DEEPSORT TRACKER ===
model = YOLO(MODEL_PATH)

tracker = DeepSort(
    max_age=7,                # for How long to keep lost tracks
    n_init=8,                 # Number of frames required to confirm a new track
    max_cosine_distance=0.22  # Appearance similarity threshold
)

# === TRACKING STATE SETUP ===
unique_ids = set()                  # Final list of unique player IDs
track_presence = defaultdict(int)   # Count how long each track has been visible
frame_count = 0

# === BUFFER TO HELP WITH CUSTOM RE-ID ===
# Each entry: (player_id, (frame_lost, bbox, centroid))
lost_tracks = deque(maxlen=100)

print("Starting Re-ID tracking with DeepSORT...")

# === FUNCTION TO COMPUTE IOU BETWEEN TWO BOXES ===
def compute_iou(box1, box2):
    # Inputs are in format: [x1, y1, x2, y2]
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxBArea = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

# === MAIN PROCESSING LOOP ===
track_id_map = {}  # Map raw tracker ID to stable

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    detections = []
    results = model(frame, verbose=False)[0]

    # === COLLECT VALID DETECTIONS ===
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if conf < CONFIDENCE_THRESHOLD or cls_id not in TRACK_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)
        if area < MIN_BOX_AREA:
            continue

        w, h = x2 - x1, y2 - y1
        detections.append(([x1, y1, w, h], conf, cls_id))

    # === UPDATE TRACKS USING DEEPSORT ===
    tracks = tracker.update_tracks(detections, frame=frame)
    active_ids = set()

    for track in tracks:
        if not track.is_confirmed():
            continue

        raw_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        centroid = ((l + r) // 2, (t + b) // 2)

        # === CUSTOM RE-ID CHECK ===
        reassigned = False
        for lost_id, (frame_lost, old_box, old_centroid) in list(lost_tracks):
            if frame_count - frame_lost > REID_FRAME_TOLERANCE:
                continue

            iou = compute_iou([l, t, r, b], old_box)
            dist = np.linalg.norm(np.array(centroid) - np.array(old_centroid))
            if iou > IOU_THRESHOLD or dist < 50:
                track_id_map[raw_id] = lost_id
                reassigned = True
                break

        assigned_id = track_id_map.get(raw_id, raw_id)
        track_presence[assigned_id] += 1

        # Only consider ID valid if it has been seen for a few frames
        if track_presence[assigned_id] < 5:
            continue

        unique_ids.add(assigned_id)
        active_ids.add(assigned_id)

        # === DRAW BOUNDING BOXES AND ID ===
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f'Player {assigned_id}', (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # === IDENTIFY LOST TRACKS AND ADD TO RE-ID BUFFER ===
    current_ids = {track.track_id for track in tracks if track.is_confirmed()}
    lost_ids = set(track_id_map.keys()).union(current_ids) - current_ids

    for lost_raw_id in lost_ids:
        assigned_id = track_id_map.get(lost_raw_id, lost_raw_id)
        if assigned_id not in active_ids:
            for track in tracks:
                if track.track_id == lost_raw_id:
                    l, t, r, b = map(int, track.to_ltrb())
                    centroid = ((l + r) // 2, (t + b) // 2)
                    lost_tracks.append((assigned_id, (frame_count, [l, t, r, b], centroid)))

    # === DISPLAY CURRENT PLAYER COUNT ===
    cv2.putText(frame, f'Unique Players: {len(unique_ids)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    out.write(frame)
    cv2.imshow("Custom Re-ID Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === CLEANUP RESOURCES ===
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"[DONE] Frames Processed: {frame_count}")
print(f"[INFO] Final Unique Player Count (Re-ID Stable): {len(unique_ids)}")
print(f"[INFO] Output saved to: {OUTPUT_PATH}")
