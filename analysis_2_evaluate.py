import os
import sys
import cv2
import math
import json
import ipdb
import dlib
import numpy as np
from PIL import Image
import alignfaces as af
import matplotlib.pyplot as plt
from shapely.geometry import Point, mapping
from shapely.geometry.polygon import Polygon

'''
	Get all files in a given folder path
'''
def files_in_folder(folder_fname):
	return [f for f in os.listdir(folder_fname) if os.path.isfile(os.path.join(folder_fname, f)) and f != ".DS_Store"]

def flip_cood(cood):
	return [[c[0],c[1]] for c in cood]

'''
	Determine the intersection over the boundary
'''
def intersection_over_boundary(boolean_map, this_boundary):
	polygon = Polygon(this_boundary["cood"])
	intersection_n = 0
	boundary_covered = 0
	for y in range(len(boolean_map)):
		for x in range(len(boolean_map[y])):
			if (polygon.contains(Point(x, y))):
				intersection_n += boolean_map[y][x]
				boundary_covered += 1
	return intersection_n, boundary_covered

# Declare the collective data for the analysis (to be populated)
image_analysis_data = list()
image_analysis_data_attachment = list()

'''
	Race and gender definitions (as taken from UTKFace - https://susanqq.github.io/UTKFace)
'''
utkface_references = {
		"genders" : {
			"0" : "Male",
			"1" : "Female"
		},
		"races" : {
			"0" : "Caucasian",
			"1" : "African",
			"2" : "Asian",
			"3" : "Indian",
			"4" : "Other" 
		}
	}

IMAGE_DATASET_CONFIG = "UTKFACE"#"IMDB_WIKI"
if (IMAGE_DATASET_CONFIG == "IMDB_WIKI"):
	PREDICTIONS_FILENAME = "metadataset_imdbwiki_predictions.json"
else:
	PREDICTIONS_FILENAME = "metadataset_utkface_predictions.json"

folder_access_name ="utkface"
if (IMAGE_DATASET_CONFIG == "IMDB_WIKI"):
	folder_access_name ="wiki-imdb"

predictions_json = json.loads(open(PREDICTIONS_FILENAME).read())["predictions"]


'''
	Here we pre-define some DLib-referenced points for facial features.
	Note: Negative values here correspond to corners of the image.
'''
boundaries_facial_template = {
	"nose" : {
		"point_order" : [21, 31, 32, 33, 34, 35, 22]
	},
	"eye_l" : {
		"point_order" : [0, 17, 18, 19, 20, 21, 27, 28, 2, 1]
	},
	"eye_r" : {
		"point_order" : [16, 15, 14, 28, 27, 22, 23, 24, 25, 26]
	},
	"cheek_l" : {
		"point_order" : [2, 28, 29, 30, 31, 48, 6, 5, 4, 3]
	},
	"cheek_r" : {
		"point_order" : [14, 13, 12, 11, 10, 54, 35, 30, 29, 28]
	},
	"eye_l_inner" : {
		"point_order" : [36, 37, 38, 39, 40, 41]
	},
	"eye_r_inner" : {
		"point_order" : [42, 43, 44, 45, 46, 47]
	},
	"mouth_inner" : {
		"point_order" : [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
	},
	"chin" : {
		"point_order" : [6, 48, 31, 32, 33, 34, 35, 54, 10, 9, 8, 7]
	},
	"top" : {
		"point_order" : [-1, 0, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 16, -2]
	},
	"left" : {
		"point_order" : [-1, 0, 1, 2, 3, 4, 5, 6, -3]
	},
	"right" : {
		"point_order" : [-2, 16, 15, 14, 13, 12, 11, 10, -4]
	},
	"bottom" : {
		"point_order" : [-3, 6, 7, 8, 9, 10, -4]
	}
}

IMAGE_BASE_DIMENSION = 224
boundaries_facial_backmap_negatives = {
	"-1" : [0,0],
	"-2" : [IMAGE_BASE_DIMENSION,0],
	"-3" : [0,IMAGE_BASE_DIMENSION],
	"-4" : [IMAGE_BASE_DIMENSION,IMAGE_BASE_DIMENSION]
}

def draw_shaded_boundary(this_image, this_boundary, alpha = 0.3):
	img_cropped_overlay = this_image.copy()
	cv2.fillPoly(img_cropped_overlay, this_boundary["cood"], this_boundary["color"])
	return cv2.addWeighted(img_cropped_overlay, alpha, this_image, 1 - alpha, 1)

# Load in the face shape predictor, as taken from http://dlib.net/files
shape_predictor_fname = "shape_predictor_68_face_landmarks.dat"
shape_predictor = dlib.shape_predictor(shape_predictor_fname)

# Load in the frontal face detector from DLib
frontal_face_detector = dlib.get_frontal_face_detector()

folder_access_name_attach = os.path.join(os.getcwd(), folder_access_name)

for f in files_in_folder(os.path.join(folder_access_name_attach, "img-cropped")):
	# If the file exists
	if ((f in predictions_json) and (os.path.isfile(os.path.join(folder_access_name_attach, "img-masks", f)))):
		f = "12_1_0_20170109214236402.jpg"
		# Create a copy of the template 'boundaries_facial' dictionary
		boundaries_facial = boundaries_facial_template.copy()
		for k in boundaries_facial.keys():
			boundaries_facial[k]["cood"] = [None for x in range(len(boundaries_facial[k]["point_order"]))]

		# Firstly retrieve the aligned and cropped image
		img_fname_cropped = os.path.join(folder_access_name_attach,"img-cropped", f)
		img_cropped = cv2.imread(img_fname_cropped)

		# Then retrieve the image mask from the LIME explanation
		img_fname_mask = os.path.join(folder_access_name_attach,"img-masks", f)
		img_mask = cv2.imread(img_fname_mask)

		# Uncomment to see mask associated with image
		#cv2.imshow(f,img_mask)
		#cv2.waitKey(0)

		'''
			Construct the Boolean map of the image mask, by manner of contrast - we are not concerned with the 
			possible of loss of data by this approach, as all masks already have high contrast.
		'''
		BOOLEAN_MAP_CONTRAST_THRESHOLD_VALUE = 200
		boolean_map = np.array(Image.open(img_fname_mask).convert('L').point(
									(lambda x : 255 if x > BOOLEAN_MAP_CONTRAST_THRESHOLD_VALUE else 0), mode='1'))
		# Adjust the values of the Boolean map for post-processing
		boolean_map = np.where(boolean_map == True, 1.0, boolean_map)
		boolean_map = np.where(boolean_map == False, 0.0, boolean_map)

		'''
			Attempt to determine the polygons that define the various facial boundaries.
		'''

		# Detect the face(s) within the image prior to enumeration.
		input_img = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
		detected = frontal_face_detector(input_img, 1)

		proceed = True
		# Then iterate over all detected faces within the image
		for k,d in enumerate(detected):
			try:
				img_mask_as_cv2 = cv2.imread(img_fname_mask)
				# For each face, determine the facial shape from the DLib shape predictor
				facial_shape = shape_predictor(input_img, d)
				# Then for each point of the face, attempt to determine the boundaries
				for i in range(facial_shape.num_parts):
					# Declare this point
					p = facial_shape.part(i)

					# Uncomment this section to see point labels
					#cv2.circle(img_cropped, (p.x, p.y), 2, 255, 1)
					#cv2.putText(img_cropped, str(i), (p.x + 4, p.y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255))

					# Attach the points for the polygons to the 'boundaries_facial' dictionary
					for feature in boundaries_facial.keys():
						if (i in boundaries_facial[feature]["point_order"]):
							boundaries_facial[feature]["cood"][boundaries_facial[feature]["point_order"].index(i)] = [p.x,p.y]

				# Attach the corners for the polygons to the 'boundaries_facial' dictionary
				for x in [-1, -2, -3, -4]:
					for feature in boundaries_facial.keys():
						if (x in boundaries_facial[feature]["point_order"]):
							index = boundaries_facial[feature]["point_order"].index(x)
							boundaries_facial[feature]["cood"][index] = boundaries_facial_backmap_negatives[str(x)]

				#Subtract the polygonal ranges for certain features that overlap
				for side in ["l", "r"]:
					feature = "cheek_%s" % (side)
					boundaries_facial[feature]["cood"] = [[int(math.floor(y)) for y in x]
						for x in mapping(Polygon(boundaries_facial[feature]["cood"]).difference(
							Polygon(boundaries_facial["nose"]["cood"])))["coordinates"][0]]
					feature = "eye_%s" % (side)
					boundaries_facial[feature]["cood"] = [[int(math.floor(y)) for y in x] 
						for x in mapping(Polygon(boundaries_facial[feature]["cood"]).difference(
							Polygon(boundaries_facial["nose"]["cood"])))["coordinates"][0]]
				

				for feature in boundaries_facial.keys():
					# Calculate the intersection percentage of the Boolean map of the image mask on the given facial feature
					boundaries_facial[feature]["intersection_n"],boundaries_facial[feature]["boundary_covered"] = intersection_over_boundary(boolean_map, boundaries_facial[feature])
					boundaries_facial[feature]["boundary_covered"] = boundaries_facial[feature]["intersection_n"]/boundaries_facial[feature]["boundary_covered"]
					boundaries_facial[feature]["intersection_n"] /= sum(sum(boolean_map))

					# Uncomment this section to visualise the contours of the facial features
					cv2.polylines(img_mask_as_cv2,[np.array(boundaries_facial[feature]["cood"], np.int32)] ,True,(255,255,255), 1)
				#cv2.imshow("",cv2.imread(img_fname_mask))
				#cv2.waitKey(0)
				#print(json.dumps(boundaries_facial,indent=3))
				cv2.imshow("",img_mask_as_cv2)
				cv2.waitKey(0)
				#ipdb.set_trace()
				sys.exit()
			except Exception as e:
				print(e)
				proceed = False
		
		if (proceed):
			try:
				# Subtract intersection values for certain facial features that reside in other facial features
				# i.e. 'mouth_inner' feature resides in 'chin' feature
				boundaries_facial["chin"]["intersection_n"] -= boundaries_facial["mouth_inner"]["intersection_n"]
				for side in ["l", "r"]:
					# i.e. 'eye_x_inner' feature resides in 'eye_x' feature
					boundaries_facial["eye_%s" % (side)]["intersection_n"] -= boundaries_facial["eye_%s_inner" % (side)]["intersection_n"]
				
				'''
					Begin the construction of the analysis' data object; each includes the following:

						- The activations of the LIME explainability package for the image segments
						- The 'ground truth' age of the individual in the image, as well as the estimated age from the classifier
						- Other metadata associated with the individual in the image (i.e. the gender and race)
					
				'''
				image_analysis_data_obj = dict()
				# Apply the activations
				for feature in boundaries_facial.keys():
					image_analysis_data_obj["activation_%s_from_all" % (feature)] = boundaries_facial[feature]["intersection_n"]
					image_analysis_data_obj["activation_%s_pct_of_feature" % (feature)] = boundaries_facial[feature]["boundary_covered"]

				image_analysis_data_obj["fname"] = f

				if (IMAGE_DATASET_CONFIG == "IMDB_WIKI"):
					pos = f.index("_")+1
					birth_year = int(f[pos:pos+4])
					year_taken = int(f[-8:-4])
					if (birth_year > 1900):
						age = abs(year_taken-birth_year)
						image_analysis_data_obj["age_real"] = age
					else:
						raise Exception("Dealing with a participant whose current age exceeds 100")
				else:
					for i in range(len(f.split("_"))):
						if (i == 0):
							image_analysis_data_obj["age_real"] = int(f.split("_")[i])
						if (i == 1):
							image_analysis_data_obj["gender_real"] = utkface_references["genders"][f.split("_")[i]]
						if (i == 2):
							image_analysis_data_obj["race_real"] = utkface_references["races"][f.split("_")[i]]

				image_analysis_data_obj["age_estimation"] = predictions_json[f]
				image_analysis_data_obj["age_estimation_accuracy"] = (100-abs(image_analysis_data_obj["age_estimation"]-image_analysis_data_obj["age_real"]))/100.0
				

				# Print the data object for a single image
				#print(json.dumps(image_analysis_data_obj,indent=3))
				

				# Append the data object to the entire analysis' collective data
				image_analysis_data.append(image_analysis_data_obj)
				image_analysis_data_attachment.append({"fname" : boundaries_facial, "boundaries_facial" : boundaries_facial})

				print("Completed analysis of file:", f)
			except Exception as e:
				print(e)
				proceed = False

'''
	Print the metadataset's image segment activations
'''
print(json.dumps(image_analysis_data,indent=3))

'''
	Print the metadataset's image segment attachment
'''
print(json.dumps(image_analysis_data_attachment,indent=3))


ipdb.set_trace()
