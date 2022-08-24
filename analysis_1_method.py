from pathlib import Path
import cv2
import imutils
import dlib
import ipdb
import time
import simplejson as json
import numpy as np
import argparse
from contextlib import contextmanager
from omegaconf import OmegaConf
from tensorflow.keras.utils import get_file
from src.factory import get_model
from keras.applications import inception_v3 as inc_net
import lime
from keras.models import Sequential
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import random
from skimage.segmentation import mark_boundaries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.models import load_model
from keras.preprocessing import image
import os,sys
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
from sklearn.preprocessing import LabelEncoder
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import io
import json
from PIL import Image as PilImage
from matplotlib.patches import Ellipse
import alignfaces as af

pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.6/EfficientNetB3_224_weights.11-3.44.hdf5"
modhash = '6d7f7b7ced093a8b3ef6399163da6ece'

def nrm(data):
	return (data - np.min(data)) / (np.max(data) - np.min(data))

'''
	Get all files in a given folder path
'''
def files_in_folder(folder_fname):
	return [f for f in os.listdir(folder_fname) if os.path.isfile(os.path.join(folder_fname, f)) and f != ".DS_Store"]

def convert_and_trim_bb(image, rect):
	# extract the starting and ending (x, y)-coordinates of the
	# bounding box
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()
	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])
	# compute the width and height of the bounding box
	w = endX - startX
	h = endY - startY
	# return our bounding box coordinates
	return (startX, startY, w, h)

def main():

	report = dict()

	img_size = 224 # This value is derived from `model_name, img_size = Path(weight_file).stem.split("_")[:2]`

	my_faces_path = os.path.join(os.getcwd(),"img","")
	dir_alignment = os.path.join(os.getcwd(),"img-aligned","")
	file_prefix = ""
	file_postfix = "jpg"

	
	'''
		Alignment Phase

		Sub-objective: In order to correctly apply the age estimator, we must align the image. In Rothe et al. (2015) Section 2.1,
		the alignment is achieved by simply rotating the image. We achieve a comparably similar result by applying the method
		of Gaspar & Garrod (2021), which rotates and warps images.

		Applies code from https://github.com/SourCherries/auto-face-align

		* Note: The source code in 'make_aligned_faces.py' has been adjusted to inhibit cropping on lines 813 - 846
		* Note: The source code in 'plot_tools.py' has been adjusted to accept all facial features (enumerated as
		['left_eyebrow', 'left_eye', 'right_eyebrow', 'right_eye', 'nose', 'mouth_outline', 'mouth_inner', 'jawline', 
		'left_iris', 'right_iris']) as the variable 'features' on line 10 & 40.
	'''
	#report["alignment"] = dict()
	'''
	# Get the facial landmarks of all images (including the jawlines)
	af.get_landmarks(my_faces_path, file_prefix, file_postfix, include_jaw=True)
	# Dis-include the files that do not pass landmark detection
	af.exclude_files_with_bad_landmarks(my_faces_path)
	# Report on the alignments
	numbers = af.landmarks_report(my_faces_path, file_prefix, file_postfix)
	report["alignment"]["images_total"] = numbers['num_total_images']
	report["alignment"]["images_failed_undetected"] = numbers['num_failed_detections']
	report["alignment"]["images_failed_detected"] = numbers['num_detected_but_removed']
	# Conduct the alignment procedure on all images
	'''
	af.align_procrustes(my_faces_path, include_features=None, color_of_result='rgb')

	'''
		Face Detection Using DLIB

		Sub-objective: In order to correctly apply the age estimator, we must crop the faces from the input files, allowing
		for a 40% margin on all sides as articulated in Section 2.1 of Rothe et al. (2015).

		Applies code from https://github.com/davisking/dlib
	'''
	dir_cropping = os.path.join(os.getcwd(),"img-cropped")
	# Create the directory for the face detections
	try:
		os.mkdir(dir_cropping)
	except:
		pass
	report["cropping"] = dict()
	report["cropping"]["images_failed_detected"] = 0
	files_in_dir_alignment = files_in_folder(dir_alignment)
	report["cropping"]["images_total"] = len(files_in_dir_alignment)
	for f in files_in_dir_alignment:
		try:
			# The image path
			img_path = os.path.join(dir_alignment, f)
			# Initiate the face detector
			detector = dlib.get_frontal_face_detector()
			# Declaring the margin size
			margin = 0.4
			# Read the image into a CV2 container
			img = cv2.imread(str(img_path), 1)
			# Determine the image's dimensions
			h, w, _ = img.shape
			'''
				TODO
				The model converts BGR to RGB prior to prediction (see line 98 of 
					https://github.com/yu4u/age-gender-estimation/blob/master/demo.py)
			'''
			input_img_BGR2RGB = img.copy()
			# Apply the detector on the input image
			boxes = [convert_and_trim_bb(input_img_BGR2RGB, detection) 
						for detection in detector(input_img_BGR2RGB, 1)]
			# Initiate the cropped faces list
			cropped_faces = list()
			# Loop over the bounding boxes, isolating the cropped faces
			for (x, y, w, h) in boxes:
				# Determine the margins on image's true dimensions
				margin_w = int(round(w*margin))
				margin_h = int(round(h*margin))
				# Execute the cropping
				this_input_image = input_img_BGR2RGB[y-margin_h:y+h+margin_h, x-margin_w:x+w+margin_w]
				# Resize the image to the dimensions accepted by the model
				this_input_image = cv2.resize(this_input_image, (img_size, img_size))
				# Add the image to array
				cropped_faces.append(this_input_image)
			# Write the DLIB-cropped image to file
			cv2.imwrite(os.path.join(dir_cropping, f),cropped_faces[0])
		except:
			# Pass failed detections
			report["cropping"]["images_failed_detected"] += 1
			pass

	# Retrieve the weight file associated with the pretrained model
	weight_file = get_file("EfficientNetB3_224_weights.11-3.44.hdf5", pretrained_model, 
		cache_subdir="pretrained_models", file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))
	# Configure the model
	model_name, img_size = Path(weight_file).stem.split("_")[:2]
	img_size = int(img_size)
	cfg = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={img_size}"])
	model = get_model(cfg)
	model.load_weights(weight_file)
	# This function is used to isolate only the result for age estimation from the model for the LIME explainer
	def classifier_fn(input_img):
		# The age estimation is stored at index 1
		results = model.predict(input_img)[1]
		return results

	'''
		Explanation Masks using LIME

		Sub-objective: In order to understand the classification of an image, the LIME explainability package (Ribeiro et al, 2016) 
		is leveraged, to understand what segments of an image contribute or detract from the classification.

		Applies code from https://lime-ml.readthedocs.io/en/latest/lime.html
	'''
	dir_masks = os.path.join(os.getcwd(),"img-masks")
	# Create the directory for the explanation masks
	try:
		os.mkdir(dir_masks)
	except:
		pass
	report["predictions"] = dict()
	report["explanations"] = dict()
	report["explanations"]["images_failed_detected"] = 0
	files_in_dir_cropping = files_in_folder(dir_cropping)
	report["explanations"]["images_total"] = len(files_in_dir_cropping)
	for f in files_in_dir_cropping:
		if ("jpg" in f):
			try:
				# The image path
				img_path = os.path.join(dir_cropping, f)
				# The image must be applied to a container for the model
				img_container = np.empty((1, img_size, img_size, 3))
				img_container[0] = cv2.imread(img_path, 1)
				# Calculate the prediction on the image
				results = model.predict(img_container)
				# The expected age is the dot product of a 100-index array by the result (see Section 2.2.2 of Rothe et al, 2015)
				ages = np.arange(0, 101).reshape(101, 1)
				predicted_age = results[1].dot(ages).flatten()[0]
				# Record the age prediction
				report["predictions"][f] = predicted_age
				# The maximum number of features we will allow for is very large
				maximum_features = 10000
				# Initialise the image explainer with 'forward selection' to correctly select the number of features for the instance
				explainer = lime_image.LimeImageExplainer(random_state=42, feature_selection="forward_selection")
				# Generate the explanation
				explanation = explainer.explain_instance(nrm(img_container[0]).astype('double'), 
											classifier_fn, top_labels=1, hide_color=0, num_samples=100)
				# Retrieve the masks for all the explanations
				_, mask = explanation.get_image_and_mask(explanation.top_labels[0], 
											positive_only=False, num_features=maximum_features, hide_rest=False)
				plt.imsave(os.path.join(dir_masks, f),mask)
			except:
				# Pass failed explanations
				report["explanations"]["images_failed_detected"] += 1

	'''
		Landmark Masks
	'''
	'''
	af.get_landmarks(os.path.join(dir_cropping,""), file_prefix, file_postfix, include_jaw=True)
	features = ['left_eyebrow', 'left_eye', 'right_eyebrow', 'right_eye',
					'nose', 'mouth_outline', 'mouth_inner', 'jawline',
					'left_iris', 'right_iris']
	with io.open(dir_cropping + '/landmarks.txt', 'r') as f:
		imported_landmarks = json.loads(f.readlines()[0].strip())
	guys = list(imported_landmarks.keys())
	num_faces = len(guys)
	for (guy, i) in zip(guys, range(num_faces)):
		gray = np.array(PilImage.open(dir_cropping + "/" + guy).convert("L"))
		this_guy = imported_landmarks[guy]
		these_x = np.empty((0, 3))
		these_y = np.empty((0, 3))
		for f in features:
			tempy = np.array(this_guy[f])
			x = tempy[::2]
			y = tempy[1::2]
			these_x = np.append(these_x, x)
			these_y = np.append(these_y, y)
		wh = gray.shape[1] / gray.shape[0]
		fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
		fig.set_size_inches(w=8*wh, h=8)
		ax.imshow(gray, cmap='gray', vmin=0, vmax=255)
		ax.plot(these_x, these_y, 'r.', linewidth=0, markersize=6)
		ax.set_xlabel(guy, fontsize=24)
		title_str = ("Face " + str(i+1) + " out of " + str(num_faces) +
					 " total." + "\n Close this window to continue.")
		plt.suptitle(title_str, fontsize=24)
		ax.axis("scaled")
		plt.show()
	'''
	print(json.dumps(report, indent=3))
	sys.exit()

if __name__ == '__main__':
	main()
