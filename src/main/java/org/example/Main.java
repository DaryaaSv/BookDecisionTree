package org.example;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.associations.Apriori;

public class Main {
    public static void main(String[] args) throws Exception {
        // Load dataset
        DataSource source = new DataSource("data.arff");
        Instances data = source.getDataSet();

        // Discretize numeric attributes
        Discretize discretize = new Discretize();
        discretize.setInputFormat(data);
        Instances discretizedData = Filter.useFilter(data, discretize);

        // Apply Apriori
        Apriori apriori = new Apriori();
        apriori.buildAssociations(discretizedData);

        // Output rules
        System.out.println(apriori);
    }
}
