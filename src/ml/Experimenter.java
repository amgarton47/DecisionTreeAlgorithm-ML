package ml;

import ml.classifiers.Classifier;
import ml.classifiers.DecisionTreeClassifer;
import ml.classifiers.RandomClassifier;

public class Experimenter {

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

	public static void main(String[] args) {
		final int NUM_TRIALS = 100;
		final int DEPTH = 10;
		final String pathToDataset = "src/data/titanic-train.csv";

		DataSet dataset = new DataSet(pathToDataset);

//		System.out.print("Random classification accuracy: ");
//		RandomClassifier random = new RandomClassifier();
//		System.out.println(Math.floor(runXTrials(dataset, random, NUM_TRIALS) * 10000) / 10000 + "%");

//		System.out.println(dt);

//		System.out.print("Decision tree classification accuracy: \n\n");
//		for (int i = 0; i <= 10; i++) {
//			System.out.print("Depth = " + i + ": ");
//			DecisionTreeClassifer dt = new DecisionTreeClassifer();
//			dt.setDepthLimit(i);
//			double[] results = runXTrials(dataset, dt, NUM_TRIALS, 0.8);
//			System.out.print(" " + results[0] + "\\% & ");
//			System.out.println(" " + results[1] + "\\%");
//		}

		System.out.print("\nDecision tree classification accuracy: \n\n");
		double splitFrac = 0.05;
		for (int i = 0; i <= 17; i++) {
			System.out.print("split fraction = " + splitFrac + ": ");
			DecisionTreeClassifer dt = new DecisionTreeClassifer();
			dt.setDepthLimit(-1);
			double[] results = runXTrials(dataset, dt, NUM_TRIALS, splitFrac);
			System.out.print(" " + results[0] + "\\% &");
			System.out.println(" " + results[1] + "\\%");
			splitFrac += 0.05;
			splitFrac = Math.floor(splitFrac * 100) / 100;
		}
	}

}
