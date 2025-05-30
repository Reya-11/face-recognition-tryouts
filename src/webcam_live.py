import cv2
import time  # Add this import
from src.align_face import align_face
from src.embedder import get_embedding
from src.classifier import load_classifier
from src.personal_memory import search_memory, add_to_memory

clf = load_classifier()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

PROCESS_EVERY_N_FRAMES = 3
frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read from webcam")
            break

        frame_count += 1

        # Only process face recognition every N frames
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            aligned = align_face(frame)
            if aligned is not None:
                emb = get_embedding(aligned)
                prob = clf.predict_proba([emb])[0]
                pred = clf.classes_[prob.argmax()]
                conf = prob.max()

                if conf < 0.7:
                    name, score = search_memory(emb)
                    if name:
                        label = f"[Memory] {name} ({score:.2f})"
                    else:
                        label = "Unknown"
                else:
                    label = f"{pred} ({conf:.2f})"
                    add_to_memory(pred, emb)

                cv2.putText(frame, label, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press ESC to quit", (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Live Recognition", frame)
        if cv2.waitKey(1) == 27:
            break
except KeyboardInterrupt:
    print("\nStopping recognition...")
finally:
    cap.release()
    cv2.destroyAllWindows()
