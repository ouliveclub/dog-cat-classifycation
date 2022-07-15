import os
import sys
import cv2 as cv
from keras.models import load_model
import numpy as np

# Đường dẫn tới file model.h5
model_path = './models/dog_cat_trained_224.h5'

# Load model
model = load_model(model_path)

# Đường dẫn tới thư mục chứa ảnh
image_path = './cat-pics/unnamed.png'

# Load ảnh
image = cv.imread(image_path)

image_size = (224, 224)

# Hình ảnh để show lên
display_image = image.copy()

# Resize về kích thước 224x224
image = cv.resize(image, image_size)

# Mở rộng thêm một chiều cho image
image = np.expand_dims(image, axis=0)

# Dự đoán
y = model.predict(image)

# Lấy nhãn
label = np.argmax(y, axis=1)

# Độ chính xác
percent = round(float(y[0][label] * 100.0), 2)

result = ''
if label == 0:
    result = 'Cat'
else:
    result = 'Dog'

message = f'{result} ({percent}%)'

# Vẽ kết quả lên ảnh
cv.rectangle(display_image, (0,0), (170, 35), (255,0,0), -1)
cv.putText(display_image, message, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv.LINE_AA)

cv.imshow('Phan loai cho meo - Live Club', display_image)

cv.waitKey(0)
cv.destroyAllWindows()