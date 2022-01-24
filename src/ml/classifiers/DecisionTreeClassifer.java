package ml.classifiers;

import java.util.ArrayList;

import ml.DataSet;
import ml.Example;

public class DecisionTreeClassifer implements Classifier {
	DecisionTreeNode root = new DecisionTreeNode(0);
	DataSet dataset = new DataSet("src/data/default.csv");
//	DataSet dataset = new DataSet("src/data/titanic-train.csv");
	int depth = 100; // = dataset.getAllFeatureIndices().size();

	public DecisionTreeClassifer() {

	}

	void setDepthLimit(int depth) {
		this.depth = depth;
	}

	public DecisionTreeNode trainRecursive(DecisionTreeNode dt, ArrayList<Example> data) {
//		if(true) {
//			return dt;
//		}
		
		// split data on "best" remaining feature
		int splitFeature = calculateScore(data);
		
		ArrayList<Example> dataLeft = new ArrayList<Example>();
		ArrayList<Example> dataRight = new ArrayList<Example>();
		
		for(Example example: data) {
			if(example.getFeature(splitFeature) == 0.0) {
				dataLeft.add(example);
			}else {
				dataRight.add(example);
			}
		}

		return null;
	}

	// calculates the scores of the remaining features
	// returns the feature that should be split on based on train error
	private int calculateScore(ArrayList<Example> data) {
		ArrayList<Double> accuracies = new ArrayList<Double>();

		for (int feature : dataset.getAllFeatureIndices()) {
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

			accuracies.add(accuracy);
			System.out.println("feature: " + feature + " error: " + accuracy);
		}

		// identifies and returns the feature with the highest accuracy
		// (lowest training error) for splitting
		double max = 0.0;
		int feature = 0;
		for (int i = 0; i < accuracies.size(); i++) {
			if (max < accuracies.get(i)) {
				max = accuracies.get(i);
				feature = i;
			}
		}
		
		return feature;
	}

	@Override
	public void train(DataSet data) {
		trainRecursive(root, dataset.getData());
	}

	@Override
	public double classify(Example example) {
		return 0;
	}

	@Override
	public String toString() {
		return root.treeString(dataset.getFeatureMap());
	}

	public static void main(String[] args) {
		DecisionTreeClassifer dt = new DecisionTreeClassifer();

//		for (Example e : dt.dataset.getData()) {
//			System.out.println(e);
//		}
	}
}
