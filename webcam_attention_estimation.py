import cv2

from sagenet import SAGENet

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize model
model_path = "models/picanet_sage_224x224/picanet_sage_224x224.onnx"
attention_estimator = SAGENet(model_path)

cv2.namedWindow("Attention heatmap", cv2.WINDOW_NORMAL)	
while cap.isOpened():

	# Read frame from the video
	ret, frame = cap.read()
	if not ret:	
		break
	
	# Estimate depth and colorize it
	heatmap = attention_estimator(frame)
	color_heatmap = attention_estimator.draw_heatmap(frame)
	
	cv2.imshow("Attention heatmap", color_heatmap)

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		cv2.imwrite("out.jpg", color_heatmap)
		break

cap.release()
cv2.destroyAllWindows()