package org.example;

import java.io.*;
import java.util.Scanner;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.DenseInstance;
import weka.classifiers.trees.J48;
import weka.core.converters.ArffLoader;

public class BookDecisionTree {
    private J48 decisionTree;
    private Instances trainingData;

    public static void main(String[] args) {
        try {
            BookDecisionTree decisionTree = new BookDecisionTree("books.arff");
            decisionTree.trainModel();
            decisionTree.printDecisionTree();

            String binding = "Hardcover";
            String multicolor = "yes";
            String genre = "romance";
            String result = decisionTree.classifyInstance(binding, multicolor, genre);
            System.out.println("Classification for instance (hardcover, yes, romance): " + result + '\n');

            decisionTree.classifyMultipleInstances();

//            System.out.println("Do you want to insert new instance to dataset?");
//
//            Scanner scanner = new Scanner(System.in);
//            String answer = scanner.nextLine();
//
//            switch (answer){
//                case "yes":
//                    System.out.println("Binding: ");
//                    String bindingInput = scanner.nextLine();
//                    System.out.println("Multicolor (yes or no): ");
//                    String multicolorInput = scanner.nextLine().toLowerCase();
//                    System.out.println("Genre: ");
//                    String genreInput = scanner.nextLine();
//
//                    decisionTree.createInstance(bindingInput, multicolorInput, genreInput);
//                    break;
//                case "no":
//                    System.exit(0);
//            }
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public BookDecisionTree(String fileName) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(fileName));
        ArffLoader.ArffReader arff = new ArffLoader.ArffReader(reader);
        trainingData = arff.getData();
        trainingData.setClassIndex(trainingData.numAttributes() - 1);
    }

    public void trainModel() throws Exception {
        decisionTree = new J48();
        String[] options = {"-U"};
        decisionTree.setOptions(options);
        decisionTree.buildClassifier(trainingData);
    }

    public void printDecisionTree() {
        System.out.println(decisionTree.toString());
    }

    public String classifyInstance(String binding, String multicolor, String genre) throws Exception {
        Instance testInstance = createInstance(binding, multicolor, genre);
        double prediction = decisionTree.classifyInstance(testInstance);
        return trainingData.classAttribute().value((int) prediction);
    }

    public void classifyMultipleInstances() throws Exception {
        System.out.println("Classifying multiple instances:");
        for (int i = 0; i < trainingData.numInstances(); i++) {
            Instance instance = trainingData.instance(i);
            double prediction = decisionTree.classifyInstance(instance);
            String predictedClass = trainingData.classAttribute().value((int) prediction);
            System.out.println("Instance " + (i + 1) + ": Predicted class - " + predictedClass);
        }
    }

    private Instance createInstance(String binding, String multicolor, String genre) {
        Instance instance = new DenseInstance(3);
        instance.setDataset(trainingData);
        instance.setValue(trainingData.attribute(0), binding);
        instance.setValue(trainingData.attribute(1), multicolor);
        instance.setValue(trainingData.attribute(2), genre);
        return instance;
    }

}

