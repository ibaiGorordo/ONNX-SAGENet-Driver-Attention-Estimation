import cv2
import pafy
import numpy as np

from sagenet import SAGENet

# Initialize video
# cap = cv2.VideoCapture("input.mp4")

videoUrl = 'https://youtu.be/MAj6y23vNuU'
videoPafy = pafy.new(videoUrl)
print(videoPafy.streams)
cap = cv2.VideoCapture(videoPafy.streams[-1].url)

start_time = 10 # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time*30)

# Initialize model
model_path = "models/sagenet_sim.onnx"
attention_estimator = SAGENet(model_path)

cv2.namedWindow("Attention heatmap", cv2.WINDOW_NORMAL)	
while cap.isOpened():

	try:
		# Read frame from the video
		ret, frame = cap.read()
		if not ret:	
			break
	except:
		continue
	
	# Estimate attention and colorize it
	heatmap = attention_estimator(frame)
	color_heatmap = attention_estimator.draw_heatmap(frame)

	combined_img = np.hstack((frame, color_heatmap))

	cv2.imshow("Attention heatmap", combined_img)

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()