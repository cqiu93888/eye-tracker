import cv2
import mediapipe as mp
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        ih, iw, _ = frame.shape
        for face_landmarks in results.multi_face_landmarks:
            iris = face_landmarks.landmark[468]  # 左眼虹膜中心
            x = int(iris.x * iw)
            y = int(iris.y * ih)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            # 中央判斷區
            cx, cy = iw // 2, ih // 2
            rx, ry = iw // 5, ih // 5
            if cx - rx < x < cx + rx and cy - ry < y < cy + ry:
                status = "Looking at Teacher"
                color = (0, 255, 0)
            else:
                status = "Not Looking"
                color = (0, 0, 255)

            cv2.putText(frame, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Eye Tracker", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    # 初始化 MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# 開始錄影
cap = cv2.VideoCapture(0)

# 初始化時間與計數器
start_time = time.time()
total_frames = 0
attentive_frames = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    ih, iw, _ = frame.shape
    total_frames += 1
    is_attentive = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            iris = face_landmarks.landmark[468]
            x = int(iris.x * iw)
            y = int(iris.y * ih)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            # 判斷是否看向中央區域
            cx, cy = iw // 2, ih // 2
            rx, ry = iw // 5, ih // 5

            if cx - rx < x < cx + rx and cy - ry < y < cy + ry:
                is_attentive = True

    # 統計專注幀數
    if is_attentive:
        status = "Looking at Teacher"
        color = (0, 255, 0)
        attentive_frames += 1
    else:
        status = "Not Looking"
        color = (0, 0, 255)

    cv2.putText(frame, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.imshow("Eye Tracker", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 鍵結束
        break

cap.release()
cv2.destroyAllWindows()

# 顯示結果
end_time = time.time()
duration = end_time - start_time
attention_rate = (attentive_frames / total_frames) * 100 if total_frames > 0 else 0

print(f"\n🎉 總追蹤時間：{duration:.1f} 秒")
print(f"👁️ 注視教師幀數：{attentive_frames} / {total_frames}")
print(f"📊 專注比例：約 {attention_rate:.1f}%")

cap.release()
cv2.destroyAllWindows()
