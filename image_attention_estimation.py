import cv2
import numpy as np
from imread_from_url import imread_from_url

from sagenet import SAGENet

model_path = "models/sagenet_sim.onnx"
attention_estimator = SAGENet(model_path)

# Read inference image
img = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/b/b3/Salt_transport_by_a_camel_train_on_Lake_Assale_%28Karum%29_in_Ethiopia.jpg")

# Estimate attention and colorize it
attention_estimator(img)
color_heatmap = attention_estimator.draw_heatmap(img)

cv2.imwrite("output.jpg", color_heatmap)

cv2.namedWindow("Attention heatmap", cv2.WINDOW_NORMAL)
cv2.imshow("Attention heatmap", color_heatmap)
cv2.waitKey(0)

