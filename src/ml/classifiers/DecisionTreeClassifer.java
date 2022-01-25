package ml.classifiers;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

import ml.DataSet;
import ml.Example;

public class DecisionTreeClassifer implements Classifier {
	private DecisionTreeNode dtc;
	private DataSet dataset;
	private int depth = -1;

	public DecisionTreeClassifer() {
	}

	@Override
	public void train(DataSet data) {
		this.dataset = data;
		dtc = trainRecursive(null, data.getData(), new HashSet<>(data.getAllFeatureIndices()), data.getData());
	}

	private DecisionTreeNode trainRecursive(DecisionTreeNode dt, ArrayList<Example> data,
			Set<Integer> remainingFeatures, ArrayList<Example> parentData) {

		if (depth != -1 && (dataset.getAllFeatureIndices().size() - remainingFeatures.size() == depth)) {
			return new DecisionTreeNode(getMajorityLabel(data));
		} else if (data.size() == 0) {
			return new DecisionTreeNode(getMajorityLabel(parentData));
		} else if (isSameLabel(data)) {
			return new DecisionTreeNode(data.get(0).getLabel());
		} else if (sameFeatures(data)) {
			return new DecisionTreeNode(getMajorityLabel(data));
		} else if (remainingFeatures.size() == 0) {
			return new DecisionTreeNode(getMajorityLabel(data));
		}

		// split data on "best" remaining feature and remove this feature from
		// remainingFeatures
		int splitFeature = calculateScore(data, new HashSet<>(remainingFeatures));
		remainingFeatures.removeIf(feature -> feature == splitFeature);

		dt = new DecisionTreeNode(splitFeature);

		ArrayList<Example> dataLeft = new ArrayList<Example>();
		ArrayList<Example> dataRight = new ArrayList<Example>();

		for (Example example : data) {
			if (example.getFeature(splitFeature) == 0.0) {
				dataLeft.add(example);
			} else {
				dataRight.add(example);
			}
		}

		dt.setLeft(trainRecursive(dt, dataLeft, new HashSet<>(remainingFeatures), data));
		dt.setRight(trainRecursive(dt, dataRight, new HashSet<>(remainingFeatures), data));
		return dt;
	}

	@Override
	public double classify(Example example) {
		return classifyRecursive(example, dtc);
	}

	private double classifyRecursive(Example example, DecisionTreeNode root) {
		if (root.isLeaf()) {
			return root.prediction();
		} else {
			if (example.getFeature(root.getFeatureIndex()) == 0.0) {
				return classifyRecursive(example, root.getLeft());
			} else {
				return classifyRecursive(example, root.getRight());
			}
		}
	}

	public void setDepthLimit(int depth) {
		this.depth = depth;
	}

	// calculates the scores of the remaining features
	// returns the feature that should be split on based on train error
	private int calculateScore(ArrayList<Example> data, Set<Integer> remainingFeatures) {
		double max = 0.0;
		int returnFeature = 0;

		for (int feature : remainingFeatures) {
			int bin0 = 0, bin1 = 0, bin00 = 0, bin11 = 0;

			for (Example example : data) {
				if (example.getFeature(feature) == 1.0) {
					if (example.getLabel() == -1.0) {
						bin0++;
					} else {
						bin1++;
					}
				} else {
					if (example.getLabel() == -1.0) {
						bin00++;
					} else {
						bin11++;
					}
				}

			}

			// calculates the score
			double correct = Math.max(bin0, bin1) + Math.max(bin00, bin11);
			double accuracy = correct / data.size();

			if (accuracy > max) {
				max = accuracy;
				returnFeature = feature;
			}
		}

		return returnFeature;
	}

	private double getMajorityLabel(ArrayList<Example> data) {
		int num0s = 0, num1s = 0;

		for (Example example : data) {
			if (example.getLabel() == -1.0) {
				num0s++;
			} else {
				num1s++;
			}
		}
		return num0s > num1s ? -1.0 : 1.0;
	}

	private boolean sameFeatures(ArrayList<Example> data) {
		for (int i = 0; i < data.size() - 1; i++) {
			if (!data.get(i).equalFeatures(data.get(i + 1))) {
				return false;
			}
		}

		return true;
	}

	// helper function for base case where all remaining data has the same label
	private boolean isSameLabel(ArrayList<Example> data) {
		double firstLabel = data.get(0).getLabel();

		for (Example example : data) {
			if (firstLabel != example.getLabel()) {
				return false;
			}
		}
		return true;
	}

	@Override
	public String toString() {
		return dtc.treeString(dataset.getFeatureMap());
	}
}
