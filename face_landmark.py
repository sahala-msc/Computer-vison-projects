import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=3,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=0),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
            )

    cv2.imshow('Face Landmarks', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
face_mesh.close()
cv2.destroyAllWindows()
