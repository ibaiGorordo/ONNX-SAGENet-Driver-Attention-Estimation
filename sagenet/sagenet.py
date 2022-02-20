import cv2
import onnx
import onnxruntime
import numpy as np

class SAGENet():

	def __init__(self, model_path):

		# Initialize model
		self.initialize_model(model_path)

	def __call__(self, image):
		return self.estimate_attention(image)

	def initialize_model(self, model_path):

		self.session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

		# Get model info
		self.get_input_details()
		self.get_output_details()

	def estimate_attention(self, image):

		input_tensor = self.prepare_input(image)

		# Perform inference on the image
		outputs = self.inference(input_tensor)

		# Process output data
		self.heatmap = self.process_output(outputs)

		return self.heatmap

	def prepare_input(self, img):

		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		self.img_height, self.img_width = img.shape[:2]

		img_input = cv2.resize(img, (self.input_width,self.input_height))

		img_input = img_input/255
		img_input = img_input.transpose(2, 0, 1)
		img_input = img_input[np.newaxis,:,:,:]        

		return img_input.astype(np.float32)

	def inference(self, input_tensor):
		# start = time.time()
		outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})[0]

		# print(time.time() - start)
		return outputs

	@staticmethod
	def process_output(output): 

		return np.squeeze(output)
		
	def draw_heatmap(self, image = None, factor = 0.5):

		heatmap_min = self.heatmap.min()
		heatmap_max = self.heatmap.max()
		norm_heatmap = 255.0 *(self.heatmap-heatmap_min)/(heatmap_max-heatmap_min)
		color_heatmap = cv2.applyColorMap(norm_heatmap.astype(np.uint8), cv2.COLORMAP_JET)

		if image is not None:
			self.img_height, self.img_width = image.shape[:2]

			# Resize and combine it with the RGB image
			color_heatmap = cv2.resize(color_heatmap, (self.img_width, self.img_height))
			color_heatmap = cv2.addWeighted(image, factor, color_heatmap, (1-factor),0)

		return color_heatmap

	def get_input_details(self):

		model_inputs = self.session.get_inputs()
		self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

		self.input_shape = model_inputs[0].shape
		self.input_height = self.input_shape[2]
		self.input_width = self.input_shape[3]

	def get_output_details(self):

		model_outputs = self.session.get_outputs()
		self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

		self.output_shape = model_outputs[0].shape
		self.output_height = self.output_shape[2]
		self.output_width = self.output_shape[3]

if __name__ == '__main__':

	from imread_from_url import imread_from_url

	model_path = "../models/sagenet_sim.onnx"

	img = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/b/b3/Salt_transport_by_a_camel_train_on_Lake_Assale_%28Karum%29_in_Ethiopia.jpg")

	attention_estimator = SAGENet(model_path)

	attention_estimator(img)
	color_heatmap = attention_estimator.draw_heatmap(img)

	cv2.namedWindow("Attention heatmap", cv2.WINDOW_NORMAL)
	cv2.imshow("Attention heatmap", color_heatmap)
	cv2.waitKey(0)


