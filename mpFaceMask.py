import cv2
from PIL import Image
import mediapipe as mp
import time
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh()
face_mesh = mp.solutions.face_mesh.FaceMesh()

# 加载图像
image = cv2.imread('/home/dell/workspace/img/中国女性，白粉红玫瑰.png')

st = time.time()

# 将图像从BGR格式转换为RGB格式
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 使用FaceMesh进行人脸关键点检测
results = face_mesh.process(image_rgb)

channels = 3  # 3个通道表示彩色图像
blank_image = np.zeros((image.shape[1], image.shape[0], channels), dtype=np.uint8)


# 绘制人脸关键点
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            cv2.circle(blank_image, (x, y), 2, (0, 255, 0), -1)
        mask = np.zeros_like(image)            
        # 将关键点转换为像素坐标
        h, w, _ = image.shape
        landmarks = []
        x_min, y_min, x_max, y_max = np.inf, np.inf, -np.inf, -np.inf
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append([x, y])
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x), max(y_max, y)
            # 在人脸关键点区域内绘制并填充多边形
        y_min, y_max = max(0, y_min - 20), max(y_max + 20, w)
        x_min, x_max = max(0, x_min - 20), max(x_max + 20, h)
        face = image[y_min:y_max, x_min:x_max]
        cv2.fillPoly(image, [np.array(landmarks)], (255, 20, 130))

new_marks = [landmarks[j]  for i in mp.solutions.face_mesh.FACEMESH_FACE_OVAL for j in i]
# cv2.fillPoly(image, [np.array(new_marks)], (255, 12, 12))
# cv2.fillConvexPoly(image, np.array(new_marks), (255, 12, 12))
for connection in mp.solutions.face_mesh.FACEMESH_FACE_OVAL:
  start_idx = connection[0]
  end_idx = connection[1]
  # cv2.line(image, landmarks[start_idx], landmarks[end_idx], (128, 12, 12), 3 )  # 轮廓线 ~



# -- 包络线算法---
image_cp = image.copy()
hull = cv2.convexHull(np.array(landmarks))
cv2.drawContours(image_cp, [hull], 0, (0, 255, 0), 2)
img_ = cv2.fillPoly(image_cp, [np.array(hull)], (255, 255, 255))
# Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR)).resize(size = (image.shape[0] // 2, image.shape[1] // 2))

end = time.time()

print(">>, ", end - st)


Image.fromarray(cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)).resize(size = (img_.shape[0] // 3, img_.shape[1] // 3))
