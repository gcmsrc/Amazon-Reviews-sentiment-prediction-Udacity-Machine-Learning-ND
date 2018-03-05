"""

#####################

This script is used to plot a standard ROC chart for
the Machine Learning Engineer Nanodegree capstone project

#####################

"""

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as numpy

class ROCChart():

	def __init__(self, y_true, y_predicted, labels, chart_title="ROC Curve"):

		"""

			Initialise ROCChart class.

			Args:
				- y_true: numpy array
				- y_predicted: list of numpy arrays
				- labels: list of strings to be used as labels
				- chart_title: string

			Returns:
				- NA

		"""

		self.y_true = y_true
		self.y_predicted = y_predicted
		self.labels = labels
		self.chart_title = chart_title
		self.auc_scores = []
		self.fpr_values = []
		self.tpr_values = []

	def plot(self):

		"""
			
			Plot ROC Curve.

		"""

		for i in range(len(self.labels)):
			self.auc_scores.append(roc_auc_score(self.y_true, self.y_predicted[i]))
			fpr, tpr, _ = roc_curve(self.y_true, self.y_predicted[i], pos_label=1.0)
			self.fpr_values.append(fpr)
			self.tpr_values.append(tpr)

		for i in range(len(self.labels)):
			print("AUC score for {} is {:.4f}".format(self.labels[i], self.auc_scores[i]))

		fig = plt.figure(figsize=(6,6))
		for i in range(len(self.labels)):
			plt.plot(self.fpr_values[i], self.tpr_values[i], label=self.labels[i])

		plt.title(self.chart_title)
		plt.xlabel("False Positive Rate")
		plt.ylabel("True Positive Rate")
		plt.legend();