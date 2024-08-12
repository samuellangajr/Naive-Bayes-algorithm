package com.example;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Evaluation;

import java.util.Random;

public class weka {

    public static void main(String[] args) {
        try {
            DataSource source = new DataSource("src\\main\\resources\\caracteristicas.arff");
            Instances data = source.getDataSet();

            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }

            Classifier classifier = new NaiveBayes();
            classifier.buildClassifier(data);

            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(classifier, data, 10, new Random(1));

            System.out.println(eval.toSummaryString("\nResultados da Avaliação com Naive Bayes\n", false));
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());
            

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
