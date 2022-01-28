package ml;

import ml.classifiers.Classifier;
import ml.classifiers.DecisionTreeClassifier;
import ml.classifiers.RandomClassifier;

/**
 * CS158-PO - Machine Learning Assignment: 02
 * 
 * A class to experiment on the accuracy of our model as well as its
 * hyper-parameters.
 * 
 * @author Aidan Garton
 *
 */
public class Experimenter {

	// runs some passed-in number of trials given a classifier model and a fraction
	// to split the training and testing data
	public static double[] runXTrials(DataSet dataset, Classifier classifier, int numTrials, double splitFraction) {
		double trainCorrect = 0.0, testCorrect = 0.0, total1 = 0.0, total2 = 0.0;
		for (int i = 0; i < numTrials; i++) {
			DataSet[] splitData = dataset.split(splitFraction);
			classifier.train(splitData[0]);

			for (Example example : splitData[1].getData()) {
				double prediction = classifier.classify(example);

				if (prediction == example.getLabel()) {
					testCorrect++;
				}
				total1++;
			}

			for (Example example : splitData[0].getData()) {
				double prediction = classifier.classify(example);

				if (prediction == example.getLabel()) {
					trainCorrect++;
				}
				total2++;
			}
		}

		double trainAccuracy = (trainCorrect / total2) * 100;
		double testAccuracy = (testCorrect / total1) * 100;

		return new double[] { Math.floor(trainAccuracy * 10000) / 10000, Math.floor(testAccuracy * 10000) / 10000 };
	}

	// starter code for testing out our model and comparing it to a random
	// classifier
	public static void main(String[] args) {
		final int NUM_TRIALS = 100;
		final String pathToDataset = "src/data/titanic-train.csv";
		final boolean USE_GINI = true;

		DataSet dataset = new DataSet(pathToDataset);

		System.out.print("Random classification accuracy: ");
		RandomClassifier random = new RandomClassifier();
		System.out.println(Math.floor(runXTrials(dataset, random, NUM_TRIALS, 0.8)[1] * 10000) / 10000 + "%");

		// varies the depth limit of the decision tree
		System.out.print("\nDecision tree classification accuracy (depth experiment): \n\n");
		for (int i = 0; i <= 10; i++) {
			System.out.print("Depth = " + i + ": ");
			DecisionTreeClassifier dt = new DecisionTreeClassifier();
			dt.setDepthLimit(i);
			dt.setGini(USE_GINI);
			double[] results = runXTrials(dataset, dt, NUM_TRIALS, 0.8);
			System.out.print(" " + results[0] + "\\% & ");
			System.out.println(" " + results[1] + "\\%");
		}

		// varies the split fractions for testing/training data
		System.out.print("\nDecision tree classification accuracy (split fraction experiment): \n\n");
		double splitFrac = 0.05;
		for (int i = 0; i <= 17; i++) {
			System.out.print("split fraction = " + splitFrac + ": ");
			DecisionTreeClassifier dt = new DecisionTreeClassifier();
			dt.setDepthLimit(-1);
			dt.setGini(USE_GINI);
			double[] results = runXTrials(dataset, dt, NUM_TRIALS, splitFrac);
			System.out.print(" " + results[0] + "\\% &");
			System.out.println(" " + results[1] + "\\%");
			splitFrac += 0.05;
			splitFrac = Math.floor(splitFrac * 100) / 100;
		}

		DecisionTreeClassifier dt = new DecisionTreeClassifier();
		dt.setDepthLimit(5);
		dt.setGini(USE_GINI);
		System.out.println(runXTrials(dataset, dt, NUM_TRIALS, 0.8)[1]);
		
//		System.out.println(dt);
	}

}
