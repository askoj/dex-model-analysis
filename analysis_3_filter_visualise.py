import json
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import numpy as np
import warnings
warnings.filterwarnings("ignore")

'''
	Declare the features that correspond to an individual's face, as well as those that correspond to outer
	boundaries of the image.
'''
GENDERS = ["Male", "Female"]
AGE_RANGES = [[0,12], [13,18], [19,25], [26,100]]
KEYS_BOUNDARIES_GENDERS = ["gender_real", "age_estimation_accuracy", "outer_boundary_left", 
							"outer_boundary_right", "outer_boundary_bottom", "outer_boundary_top", "face"]
KEYS_RACE_AGE = ["age_estimation_accuracy", "race_real", "age_real", "age_estimation"]
KEYS_GENDER_AGE = ["age_estimation_accuracy", "gender_real", "age_real", "age_estimation"]
KEYS_RACE_AGE_BOUNDARIES = ["age_estimation_accuracy", "age_estimation", "race_real", "age_real", "face", "fname",
			 					"outer_boundary_left", "outer_boundary_right", "outer_boundary_top", "outer_boundary_bottom"]
KEYS_BOUNDARIES = ["face", "outer_boundary_left", "outer_boundary_right", "outer_boundary_top", "outer_boundary_bottom"]

RACES = ["Caucasian", "Asian", "African", "Indian", "Other"]

FEATURES_ALL = [ "activation_nose_pct_of_feature", "activation_eye_l_pct_of_feature", 
					"activation_eye_r_pct_of_feature", "activation_cheek_l_pct_of_feature", 
					"activation_cheek_r_pct_of_feature", "activation_eye_l_inner_pct_of_feature", 
					"activation_eye_r_inner_pct_of_feature", "activation_mouth_inner_pct_of_feature", 
					"activation_chin_pct_of_feature", "activation_top_pct_of_feature", 
					"activation_left_pct_of_feature", "activation_right_pct_of_feature", 
					"activation_bottom_pct_of_feature"]

FEATURES_FACIAL = [ "activation_nose_pct_of_feature", "activation_eye_l_pct_of_feature", 
					"activation_eye_r_pct_of_feature", "activation_cheek_l_pct_of_feature", 
					"activation_cheek_r_pct_of_feature", "activation_eye_l_inner_pct_of_feature", 
					"activation_eye_r_inner_pct_of_feature", "activation_mouth_inner_pct_of_feature", 
					"activation_chin_pct_of_feature"]

FEATURES_OUTER_BOUNDARIES = [x for x in FEATURES_ALL if (x not in FEATURES_FACIAL)]

'''
	This function generates some important statistics used to determine the significance of the
	results reported for this analysis, and is contextualised by race, age, and gender.
'''
def statistic_details(utkface_data, arg_race, arg_age_min, arg_age_max, arg_gender, alpha=1e-3, visualise=False):
	# An acceptable alpha has a 0.05 reading...
	# Contextualise the distribution by the parametrics
	contextualised_distribution = [round(x["age_estimation"]) for x in utkface_data 
									   if (x["race_real"] == arg_race) 
									   and (x["age_real"] >= arg_age_min)
									   and (x["age_real"] <= arg_age_max)
									   and (x["gender_real"] == arg_gender)]
	 # D'Agostino's K-squared test
	skewtest_passed = True
	k2, p = None, None
	chisquare_st = None, None
	try:
		k2, p = scipy.stats.normaltest(contextualised_distribution, axis=0)
		chisquare_st = scipy.stats.chisquare(contextualised_distribution)
	except ValueError:
		skewtest_passed = False
	result = {
		"age_min" : arg_age_min,
		"age_max" : arg_age_max,
		"race" : arg_race,
		"gender" : arg_gender,
		"mean" : np.mean(contextualised_distribution),
		"standard_deviation" : np.std(contextualised_distribution),
		"normality_test_p_value" : p,
		"normality_test_statistic" : k2,
		"n_participants" : len(contextualised_distribution),
		"kurtosis_test_n_greater_than_or_equal_to_20" : (len(contextualised_distribution) >= 20),
		"normality_test_passed" : ((p is not None) and (p < alpha)),
		"skew_test_samples_greater_than_or_equal_to_8" : skewtest_passed,
		"k_test_statistic" : chisquare_st[0],
		"k_test_p" : chisquare_st[1]
	}
	if (visualise):
		sns.displot(contextualised_distribution, binwidth=1)
	return result

'''
	Calculate the statistic details for an age range
'''
def statistic_details_for_age_range(utkface_data, this_age_min_max, alpha):
	statistic_details_all = list()
	for this_gender in GENDERS:
		for this_race in RACES:
			this_statistic_detail = statistic_details(utkface_data, this_race, this_age_min_max[0], this_age_min_max[1], this_gender, alpha)
			statistic_details_all.append(this_statistic_detail)
	return pd.DataFrame(data=statistic_details_all, columns=statistic_details_all[0].keys())

'''
	This function normalises the size of a cohort to reflect a desired number of participants.
	For especially small cohorts, a warning is raised.
'''
def normalise_cohort(cohort,desired_n_participants, verbose=False):
	if ((len(cohort) < 30) and (verbose)):
		print("WARNING: Cohort is small and may not be representative.")
	new_cohort = list()
	while(len(new_cohort) < desired_n_participants):
		new_cohort.extend(cohort)
	return new_cohort[:desired_n_participants]

'''
	This function filters a dataset (as a dictionary) by a given age range and retained keys, 
	and then separates it by a key of given categories
'''
def filter_dataset(dataset, retain_keys, key_of_categories, categorical_values, age_min=0, age_max=100):
	'''
		Firstly clean and filter the data according to age, and then separate it by the designated 
		key of categories
	'''
	cleaned_data = [{k:v for k,v in x.items() if (k in retain_keys)} for x in dataset]
	dataset_processed = dict()
	for category in categorical_values:
		dataset_processed[category] = [x for x in cleaned_data 
				if (x[key_of_categories] == category) and 
								  ((x["age_real"] >= age_min) and (x["age_real"] <= age_max))]
	cohort_sizes = {x:len(dataset_processed[x]) for x in dataset_processed.keys()}
	'''
		All cohorts are normalised to have the same number of participants as the maximum cohort
	'''
	for key in categorical_values:
		dataset_processed[key] = normalise_cohort(dataset_processed[key], max(cohort_sizes.values()))
	'''
		Return the dataset in dictionary, list, and pandas dataframe form,
		and also return the key_of_categories value to the visualisation function, as
		well as the cohort_sizes dictionary for representation
	'''
	dataset_processed_list = [x for y in dataset_processed.values() for x in y]
	return (dataset_processed, dataset_processed_list, pd.DataFrame(dataset_processed_list), 
				key_of_categories, cohort_sizes)
	
'''
	Visualise a dataset as a pairplot, and include relevant notes to the analysis
'''
def visualise_pairplot(dataset_tuple, accuracy_cutoff=0.9, verbose=False):
	dataset_dict, dataset_list, dataset_pd, key_of_categories, cohort_sizes = dataset_tuple
	'''
		Print how much of the distribution of each separated category lies on either side of the 
		accuracy cut-off value
	'''
	for key in dataset_dict.keys():
		if (verbose):
			print(key)
		'''
			Notify us of the cohort sizes as a heuristic for representativeness; we will evaluate this
			further during the analysis, however we set an indication here for convenience
		'''
		if (verbose):
			print("\tCohort size:", cohort_sizes[key])
		accuracy_cutoff_list = [x["age_estimation_accuracy"] 
								for x in dataset_list if (x[key_of_categories] == key)]
		if (verbose):
			for ranges in [[0, accuracy_cutoff], [accuracy_cutoff, 1]]:
				print("\tage_estimation_accuracy: %s - %s:" % (round(ranges[0]*100), round(ranges[1]*100)), 
					  len([x for x in accuracy_cutoff_list 
						   if (x >= ranges[0] and x <= ranges[1])])/len(dataset_dict[key]))
	return dataset_pd#sns.pairplot(dataset_pd, hue=key_of_categories)

'''
	Visualise image segmentation data as heatmaps, with correlation matices, and an isolated row of a pairplot
'''
def visualise_heatmap_corr_pairplot(dataset_tuple, image_masks_fpath, x_vars=None, pairplot_scatter_alpha=0.005, age_range=str(), this_type=str()):
	BOOLEAN_MAP_CONTRAST_THRESHOLD_VALUE = 200
	dataset_dict, dataset_list, dataset_pd, key_of_categories, cohort_sizes = dataset_tuple
	plot_kws = {
		"line_kws" : {
			"alpha" : 0
		},
		"scatter_kws" : {
			"alpha" : pairplot_scatter_alpha
		}
	}
	retained_keys = [ "face", "age_estimation_accuracy", "outer_boundary_left", "outer_boundary_right",
															"outer_boundary_top",  "outer_boundary_bottom"]
	corr_xticklabels = ["age_estimation_accuracy"]
	corr_yticklabels = ["face", "outer_boundary_left", "outer_boundary_right", "outer_boundary_top", 
																				"outer_boundary_bottom"]
	#fig, axs = plt.subplots(1, 5, figsize=(20, 20))
	#fig.set_size_inches(18.5, 10.5)
	'''
		Visualise the image segments heatmaps by race
	'''
	ii = 0
	heatmaps = dict()
	race_corr_m = list()
	for race in dataset_dict.keys():
		#print(race)
		'''
			Notify us of the cohort sizes as a heuristic for representativeness; we will evaluate this
			further during the analysis, however we set an indication here for convenience
		'''
		#print("\tCohort size:", cohort_sizes[race])

		dataset_new = [x for x in dataset_list if (x["race_real"] == race)]
		'''
			Visualise the correlational matrices on the image segments for the heatmap, as grouped by race
		'''
		
		corr_m = pd.DataFrame(pd.DataFrame(dataset_new)[retained_keys]).corr(method="pearson")[["age_estimation_accuracy"]]
		corr_m = corr_m.drop(["age_estimation_accuracy"])
		# Apply the correlational matrices as secondary heatmaps
		#labels = np.array([round(v,5) for v in list(corr_m.values.flatten())]).reshape(5, 1)
		#sns.heatmap(corr_m, ax=axs[ii], cmap='RdYlGn', annot=labels, vmin=min(corr_m.values)[0], 
		#				vmax=max(corr_m.values)[0], xticklabels=corr_xticklabels, yticklabels=corr_yticklabels)
		race_age_corr = (dict(corr_m["age_estimation_accuracy"]))
		race_age_corr["race"] = race
		race_age_corr["age_range"] = age_range
		race_age_corr["classification_type"] = this_type
		race_corr_m.append(race_age_corr)
		ii += 1

	#fig.tight_layout()
	'''
		Visualise the pairplots that accompany the correlational matrices
	'''
	
	#g = sns.pairplot(dataset_pd, x_vars=x_vars, y_vars=['age_estimation_accuracy'], kind='reg', plot_kws=plot_kws)
	# Set appropriate x limits on the pairplots
	#for x in range(5):
	#	g.axes[0,x].set_xlim((0,1))
	
	return race_corr_m



'''
	Visualise image segmentation data as heatmaps, with correlation matices, and an isolated row of a pairplot
'''
def visualise_heatmap_corr_pairplot_all(dataset_tuple, image_masks_fpath, x_vars=None, pairplot_scatter_alpha=0.005, dataset_tuple2=None, this_type="accurate", verbose=False):
	BOOLEAN_MAP_CONTRAST_THRESHOLD_VALUE = 200
	dataset_dict, dataset_list, dataset_pd, key_of_categories, cohort_sizes = dataset_tuple
	dataset_dict2, dataset_list2, dataset_pd2, key_of_categories2, cohort_sizes2 = dataset_tuple2
	plot_kws = {
		"line_kws" : {
			"alpha" : 0
		},
		"scatter_kws" : {
			"alpha" : pairplot_scatter_alpha
		}
	}
	retained_keys = [ "face", "age_estimation_accuracy", "outer_boundary_left", "outer_boundary_right",
															"outer_boundary_top",  "outer_boundary_bottom"]
	corr_xticklabels = ["age_estimation_accuracy"]
	corr_yticklabels = ["face", "outer_boundary_left", "outer_boundary_right", "outer_boundary_top", 
																				"outer_boundary_bottom"]
	fig, axs = plt.subplots(1, 5)
	fig.set_size_inches(18.5, 10.5)
	'''
		Visualise the image segments heatmaps by race
	'''
	ii = 0
	heatmaps = dict()
	race_corr_m = dict()
	for race in dataset_dict.keys():
		if (verbose):
			print(race)
		'''
			Notify us of the cohort sizes as a heuristic for representativeness; we will evaluate this
			further during the analysis, however we set an indication here for convenience
		'''
		if (verbose):
			print("\tCohort size:", cohort_sizes[race])
		# Construct each heatmap as an np.array
		heatmap_list = list()
		for dataset_list_i in [dataset_list, dataset_list2]:
			heatmap = None
			dataset_new = [x for x in dataset_list_i if (x["race_real"] == race)]
			for f in [x["fname"] for x in dataset_new]:
				boolean_heatmap = np.array(Image.open(image_masks_fpath+f).convert('L').point(
					(lambda x : 255 if x > BOOLEAN_MAP_CONTRAST_THRESHOLD_VALUE else 0), mode='1'))
				boolean_heatmap = np.where(boolean_heatmap == True, 1.0, boolean_heatmap)
				boolean_heatmap = np.where(boolean_heatmap == False, -1.0, boolean_heatmap)
				if (heatmap is None):
					heatmap = boolean_heatmap
				else:
					heatmap += boolean_heatmap
			heatmap_list.append(heatmap)
		# Apply the heatmaps to the figure
		for q in range(2):
			heatmap_list[q] *= 2/heatmap_list[q].max()
			heatmap_list[q] -= 1
		if (this_type == "accurate"):
			heatmaps[race] = heatmap_list[0] - heatmap_list[1]
		elif (this_type == "inaccurate"):
			heatmaps[race] = heatmap_list[1] - heatmap_list[0]
		elif (this_type == "observing"):
			heatmaps[race] = heatmap_list[1] + heatmap_list[0]
		heatmaps[race] *= 2/heatmaps[race].max()
		heatmaps[race] -= 1
		axs[ii].imshow(heatmaps[race], cmap='jet', interpolation='gaussian')
		axs[ii].set_title(race)
		#axs[ii].set_title(race)
		axs[ii].set_xticks([])
		axs[ii].set_yticks([])
		axs[ii].set_axis_off()
		#axs[ii].set_title(race)
		fig.savefig('%s.png' % (ii), bbox_inches=axs[ii].get_window_extent().transformed(fig.dpi_scale_trans.inverted()))
		'''
			Visualise the correlational matrices on the image segments for the heatmap, as grouped by race
		'''
		'''
		corr_m_list = list()
		for dataset_list_i in [dataset_list, dataset_list2]:
			dataset_new = [x for x in dataset_list_i if (x["race_real"] == race)]
			corr_m_list.append(pd.DataFrame(pd.DataFrame(dataset_new)[retained_keys]).corr(method="pearson")[["age_estimation_accuracy"]])
		corr_m = corr_m_list[0] - corr_m_list[1]
		corr_m = corr_m.drop(["age_estimation_accuracy"])
		# Apply the correlational matrices as secondary heatmaps
		labels = np.array([round(v,5) for v in list(corr_m.values.flatten())]).reshape(5, 1)
		sns.heatmap(corr_m, ax=axs[1,ii], cmap='RdYlGn', annot=labels, vmin=min(corr_m.values)[0], 
						vmax=max(corr_m.values)[0], xticklabels=corr_xticklabels, yticklabels=corr_yticklabels)
		'''
		ii += 1
		#race_corr_m[race] = corr_m
	fig.tight_layout()
	'''
		Visualise the pairplots that accompany the correlational matrices
	'''
	'''
	g = sns.pairplot(dataset_pd, x_vars=x_vars, y_vars=['age_estimation_accuracy'], kind='reg', plot_kws=plot_kws)
	# Set appropriate x limits on the pairplots
	for x in range(5):
		g.axes[0,x].set_xlim((0,1))
	'''
	return heatmaps

'''
	Visualise the correlations between the gender key of a dataset, and the other keys therein
'''
def visualise_corr_gender(dataset):
	matrix = (pd.DataFrame(dataset)
				[KEYS_BOUNDARIES_GENDERS].replace("Female",0).replace("Male",1).corr())
	return matrix

'''
	This function generates the necessary processed datasets from the source dataset in order to undertake the 
	necessary analyses
'''
def generate_processed_datasets(source_dataset):
	'''
		Declare the dataset that describes the 'face vs. outer boundary' image segments,
		and groups them by 'positive' or 'negative' result
	'''
	image_segment_data_face_vs_outer = { "positive" : list(), "negative" : list() }
	for elem in source_dataset:
		calculated_fields = {
			"face" : (sum([elem[x] for x in FEATURES_FACIAL])/len(FEATURES_FACIAL)),
			"outer_boundary_left" : elem["activation_left_pct_of_feature"],
			"outer_boundary_right" : elem["activation_right_pct_of_feature"],
			"outer_boundary_top" : elem["activation_top_pct_of_feature"],
			"outer_boundary_bottom" : elem["activation_bottom_pct_of_feature"] 
		}
		for k in calculated_fields.keys():
			calculated_fields[k] /=  sum(calculated_fields.values())
		elem.update(calculated_fields)
		if (elem["age_estimation_accuracy"] > 0.90):
			image_segment_data_face_vs_outer["positive"].append(elem)
		else:
			image_segment_data_face_vs_outer["negative"].append(elem)

	'''
		Declare the dataset that describes total image segment contributions to classification
	'''
	image_segment_data_contribution_pct = dict()
	for f in FEATURES_ALL:
		if (f not in image_segment_data_contribution_pct.keys()):
			image_segment_data_contribution_pct[f] = 0
		for elem in source_dataset:
			image_segment_data_contribution_pct[f] += elem[f]
		image_segment_data_contribution_pct[f] /= len(source_dataset)
	return image_segment_data_face_vs_outer, image_segment_data_contribution_pct
