package ml.classifiers;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

import ml.DataSet;
import ml.Example;

/**
 * CS158-PO - Machine Learning: Assignment 02
 * 
 * A decision tree classifier for making predictions after training on labeled
 * data.
 * 
 * @author aidangarton
 *
 */
public class DecisionTreeClassifier implements Classifier {
	private DecisionTreeNode dtc;
	private DataSet dataset;
	private int depth = -1;
	private boolean useGini = false;

	public DecisionTreeClassifier() {
	}

	@Override
	public void train(DataSet data) {
		this.dataset = data;
		dtc = trainRecursive(null, data.getData(), new HashSet<>(data.getAllFeatureIndices()), data.getData());
	}

	/**
	 * The main DT algorithm.
	 * 
	 * This function recursively builds a decision tree according to the highest
	 * scores of the features at each level of the tree. Non-leaf nodes represent
	 * and encode features of the data and leaf nodes represent a prediction for a
	 * piece of data.
	 * 
	 * This function must be passed an initial root node, the list of examples to
	 * train on, a set of the remaining features needed to be encoded, and the data
	 * of the parent node to help in one of the base cases.
	 *
	 * @param dt                - root node to build tree from
	 * @param data              - the list of examples left to train on
	 * @param remainingFeatures - the remaining features of the data to be encoded
	 * @param parentData        - the list of examples the parent trained on
	 * @return - the root node of a decision tree classifier
	 */
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
		int splitFeature;

		if (useGini) {
			splitFeature = getFeatureByGini(data, remainingFeatures);
		} else {
			splitFeature = getFeatureByTrainError(data, remainingFeatures);
		}
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

		// recursively create the left and right sub trees for the current node
		dt.setLeft(trainRecursive(dt, dataLeft, new HashSet<>(remainingFeatures), data));
		dt.setRight(trainRecursive(dt, dataRight, new HashSet<>(remainingFeatures), data));
		return dt;
	}

	/**
	 * Extra credit: Implementation of information gain using gini impurity. Returns
	 * the "best" feature to split on according to the gini method.
	 * 
	 * @param data     - data remaining
	 * @param features -
	 * @return
	 */
	private int getFeatureByGini(ArrayList<Example> data, Set<Integer> features) {
		int retFeature = (int) features.toArray()[0];
		for (int feature : features) {
			double maxImpurity = 1;

			ArrayList<Example> left = new ArrayList<Example>();
			ArrayList<Example> right = new ArrayList<Example>();

			// split the data on feature
			for (Example ex : data) {
				if (ex.getFeature(feature) == 0.0) {
					left.add(ex);
				} else {
					right.add(ex);
				}
			}

			// count label0 for left subtree
			int count1left = 0;
			for (Example ex : left) {
				if (ex.getFeature(feature) == 0.0) {
					count1left++;
				}
			}

			// count label0 for left subtree
			int count1right = 0;
			for (Example ex : right) {
				if (ex.getFeature(feature) == 0.0) {
					count1right++;
				}
			}

			// get proportion of label in left/right subtrees
			double p1, p2;

			if (left.size() == 0) {
				p1 = 0;
			} else {
				p1 = count1left / left.size();
			}

			if (right.size() == 0) {
				p2 = 0;
			} else {
				p2 = count1right / right.size();
			}

			// calculate gini index for both subtrees
			double giniLeft = 1 - p1 * p1 - (1 - p1) * (1 - p1);
			double giniRight = 1 - p2 * p2 - (1 - p2) * (1 - p2);

			// weight scores by proportion of data in each tree
			giniLeft *= (left.size() / data.size());
			giniRight *= (right.size() / data.size());

			double gini = giniLeft + giniRight;

			// set retTeature to feature that yields lowest impurity index
			if (gini < maxImpurity) {
				retFeature = feature;
			}

		}
		return retFeature;
	}

	@Override
	public double classify(Example example) {
		return classifyRecursive(example, dtc);
	}

	/**
	 * The main classification algorithm.
	 * 
	 * Recursively traverses the tree, going down left/right subtrees according to
	 * the value of the provided example's features. Will traverse right subtree if
	 * non-zero valued feature, traverses left subtree otherwise
	 * 
	 * @param example - data example to classify
	 * @param root    - the decision tree model
	 * @return - the classifaction of example
	 */
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

	/**
	 * Simple setter method for the depth limit of the decision tree.
	 * 
	 * @param depth - the desired depth limit
	 */
	public void setDepthLimit(int depth) {
		this.depth = depth;
	}

	/**
	 * Simple setter method for whether or not to use gini as splitting criterion
	 * 
	 * @param gini - true sets gini on and false sets gini off
	 */
	public void setGini(boolean gini) {
		useGini = gini;
	}

	/**
	 * Calculates the scores of the remaining features and returns the feature that
	 * should be split on based on train error
	 * 
	 * @param data              - the data left to train
	 * @param remainingFeatures - the remaining features for score to be calculated
	 *                          for
	 * @return - the index of the feature with the highest score (lowest train
	 *         error)
	 */
	private int getFeatureByTrainError(ArrayList<Example> data, Set<Integer> remainingFeatures) {
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

			// gets max score among all feature scores
			if (accuracy > max) {
				max = accuracy;
				returnFeature = feature;
			}
		}
		return returnFeature;
	}

	/**
	 * Calculates the majority label of a provided list of examples
	 * 
	 * @param data - the data to be examined
	 * @return - 1.0 or -1.0 (the majority label)
	 */
	private double getMajorityLabel(ArrayList<Example> data) {
		int num0s = 0, num1s = 0;

		for (Example example : data) {
			if (example.getLabel() == -1.0) {
				num0s++;
			} else {
				num1s++;
			}
		}
		return num0s > num1s ? -1.0 : 1.0; // oooh, fancy ternary
	}

	/**
	 * Checks if all features in a list of examples are the same.
	 * 
	 * @param data - the data to be examined
	 * @return - true if all features are same for provided data, false if otherwise
	 */
	private boolean sameFeatures(ArrayList<Example> data) {
		for (int i = 0; i < data.size() - 1; i++) {
			if (!data.get(i).equalFeatures(data.get(i + 1))) {
				return false;
			}
		}
		return true;
	}

	/**
	 * Helper function for base case where all remaining data has the same label
	 * 
	 * @param data - the data to be examined
	 * @return - true if remaining data all has the same class label, false if
	 *         otherwise
	 */
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

	// simple code to test our implementation
	public static void main(String[] args) {
		final String pathToDataset = "src/data/default.csv";
//		final String pathToDataset = "src/data/titanic-train.csv";
		DataSet dataset = new DataSet(pathToDataset);

		DecisionTreeClassifier dt = new DecisionTreeClassifier();
		dt.setDepthLimit(-1);
		dt.setGini(false);

		dt.train(dataset);
		System.out.println(dt);

		for (int i = 0; i < dataset.getData().size(); i++) {
			double pred = dt.classify(dataset.getData().get(i));

			System.out.println(dataset.getData().get(i).getLabel() == pred);
		}
	}
}
