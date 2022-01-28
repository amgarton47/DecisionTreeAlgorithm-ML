package ml;

import ml.classifiers.Classifier;
import ml.classifiers.DecisionTreeClassifier;
import ml.classifiers.RandomClassifier;

/**
 * CS158-PO - Machine Learning Assignment: 02
 * 
 * A class to experiment on the accuracy of a classification model as well as
 * its hyper-parameters.
 * 
 * @author Aidan Garton
 *
 */
public class Experimenter {

	// runs some passed-in number of trials given a classification model and a
	// fraction
	// for which to split the training and testing data
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
		System.out.print("\nDecision tree classification accuracy (depth experiment): \n");
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
		System.out.print("\nDecision tree classification accuracy (split fraction experiment): \n");
		double splitFrac = 0.05;
		for (int i = 0; i <= 17; i++) {
			System.out.print("split fraction = " + splitFrac + ": ");

			DecisionTreeClassifier dt = new DecisionTreeClassifier();
			dt.setDepthLimit(-1);
			dt.setGini(USE_GINI);

			double[] results = runXTrials(dataset, dt, NUM_TRIALS, splitFrac);
			System.out.print(" " + results[0] + "%");
			System.out.println(" " + results[1] + "%");

			splitFrac += 0.05;
			splitFrac = Math.floor(splitFrac * 100) / 100;
		}

		// varies the split fractions for testing/training data
		System.out.print("\nDecision tree classification accuracy (gini vs. training error): \n");
		for (int i = 0; i <= 10; i++) {
			DecisionTreeClassifier dtGini = new DecisionTreeClassifier();
			dtGini.setDepthLimit(i);
			dtGini.setGini(true);

			DecisionTreeClassifier dtTrainError = new DecisionTreeClassifier();
			dtTrainError.setDepthLimit(i);
			dtTrainError.setGini(false);

			double[] results1 = runXTrials(dataset, dtGini, NUM_TRIALS, 0.8);
			double[] results2 = runXTrials(dataset, dtTrainError, NUM_TRIALS, 0.8);

			System.out.print("Depth = " + i + ": ");
//			System.out.print(" " + results1[0] + "\\% &");
			System.out.print(" " + results1[1] + "% ");

//			System.out.print(" " + results2[0] + "\\% &");
			System.out.println(" " + results2[1] + "%");
		}

		// using optimal hyper-parameters (for titanic-train.csv)
		DecisionTreeClassifier dt = new DecisionTreeClassifier();
		dt.setDepthLimit(5);
		dt.setGini(USE_GINI);
		System.out.println("\n" + runXTrials(dataset, dt, NUM_TRIALS, 0.8)[1] + "%");

//		System.out.println(dt);
	}
}
