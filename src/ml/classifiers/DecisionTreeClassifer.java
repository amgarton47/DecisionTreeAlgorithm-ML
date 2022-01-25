package ml.classifiers;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

import ml.DataSet;
import ml.Example;

public class DecisionTreeClassifer implements Classifier {
	public DecisionTreeNode dtc;
	private DataSet dataset = new DataSet("src/data/default.csv");
//	DataSet dataset = new DataSet("src/data/titanic-train.csv");
	private int depth = 100; // = dataset.getAllFeatureIndices().size();

	public DecisionTreeClassifer() {

	}

	void setDepthLimit(int depth) {
		this.depth = depth;
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

	public DecisionTreeNode trainRecursive(DecisionTreeNode dt, ArrayList<Example> data,
			Set<Integer> remainingFeatures) {

		System.out.println("start");
		if (isSameLabel(data)) {
			System.out.println("bc1 reached: " + data.get(0).getLabel());
			return new DecisionTreeNode(data.get(0).getLabel());
		}

		// split data on "best" remaining feature and remove this feature from
		// remainingFeatures
		int splitFeature = calculateScore(data, new HashSet<>(remainingFeatures));
		remainingFeatures.removeIf(feature -> feature == splitFeature);

		System.out.println("feature " + splitFeature);
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
		
		dt.setLeft(trainRecursive(dt, dataLeft, new HashSet<>(remainingFeatures)));
		dt.setRight(trainRecursive(dt, dataRight, new HashSet<>(remainingFeatures)));

		return dt;
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

			System.out.println("feature: " + feature + " accuracy: " + accuracy);
		}

		return returnFeature;
	}

	@Override
	public void train(DataSet data) {
//		Set<Integer> features = data.getAllFeatureIndices();
//
//		int startingFeature = calculateScore(data.getData(), features);
//		features.removeIf(feature -> feature == startingFeature);
//		Set<Integer> remaining = features;

//		DecisionTreeNode root = new DecisionTreeNode(0);
		dtc = trainRecursive(null, dataset.getData(), new HashSet<>(dataset.getAllFeatureIndices()));
	}

	@Override
	public double classify(Example example) {
		return 0;
	}

	@Override
	public String toString() {
		return dtc.treeString(dataset.getFeatureMap());
	}

	public DecisionTreeNode ret() {
		return dtc;
	}

	public static void main(String[] args) {
		DecisionTreeClassifer dt1 = new DecisionTreeClassifer();

//		for (Example e : dt.dataset.getData()) {
//			System.out.println(e);
//		}
		DataSet dataset = new DataSet("src/data/default.csv");
		dt1.train(dataset);
		DecisionTreeNode c = dt1.ret();
		System.out.println(c.getFeatureIndex());
//		System.out.println(c.getLeft().getLeft().getLeft().getLeft());
//		System.out.println(c.getLeft().getLeft().getLeft().getLeft().getLeft());
//		System.out.println(c.getLeft().getRight());
//		System.out.println(c.getRight());
		System.out.println(dt1);

	}
}
