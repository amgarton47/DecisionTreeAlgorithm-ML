package ml;

import ml.classifiers.Classifier;
import ml.classifiers.DecisionTreeClassifer;
import ml.classifiers.RandomClassifier;

public class Experimenter {

	public static double runXTrials(DataSet dataset, Classifier classifier, int numTrials) {
		double correct = 0.0, total = 0.0;
		for (int i = 0; i < numTrials; i++) {
			DataSet[] splitData = dataset.split(0.8);
			classifier.train(splitData[0]);

			for (Example example : splitData[1].getData()) {
				double prediction = classifier.classify(example);

				if (prediction == example.getLabel()) {
					correct++;
				}
				total++;
			}
		}

		double accuracy = (correct / total) * 100;
		return accuracy;
	}

	public static void main(String[] args) {
		final int NUM_TRIALS = 100;
		final int DEPTH = 4;
		
		String pathToDataset = "src/data/titanic-train.csv";
		DataSet dataset = new DataSet(pathToDataset);

		System.out.print("Random classification accuracy: ");
		RandomClassifier random = new RandomClassifier();
		System.out.println(Math.floor(runXTrials(dataset, random, NUM_TRIALS) * 10000) / 10000 + "%");

		System.out.print("\nDecision tree classification accuracy: ");
		DecisionTreeClassifer dt = new DecisionTreeClassifer();
		dt.setDepthLimit(DEPTH);
		System.out.println(Math.floor(runXTrials(dataset, dt, NUM_TRIALS) * 10000) / 10000 + "%");
//		System.out.println(dt);
	}

}
