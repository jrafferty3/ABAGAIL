package opt.test;

import func.nn.feedfwd.FeedForwardNetwork;
import func.nn.feedfwd.FeedForwardNeuralNetworkFactory;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import shared.DataSet;
import shared.ErrorMeasure;
import shared.Instance;
import shared.SumOfSquaresError;
import shared.filt.RandomOrderFilter;
import shared.filt.TestTrainSplitFilter;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;


public class AdultRHCRestartsTest {
    private static Instance[] instances = initializeInstances();

    private static int inputLayer = 4, outputLayer = 1, trainingIterations = 1000;
    private static FeedForwardNeuralNetworkFactory factory = new FeedForwardNeuralNetworkFactory();
    private static ErrorMeasure measure = new SumOfSquaresError();
    private static DataSet set = new DataSet(instances);
    

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        List<Double> trainErrors = new ArrayList<>();
        List<Double> testErrors = new ArrayList<>();

        for (int k = 0; k < 25; k++) {
            new RandomOrderFilter().filter(set);
            TestTrainSplitFilter ttsf = new TestTrainSplitFilter(70);
            ttsf.filter(set);
            DataSet train = ttsf.getTrainingSet();
            DataSet test = ttsf.getTestingSet();
            
            FeedForwardNetwork network = factory.createClassificationNetwork(new int[] {inputLayer, 14, 14, 14, 14, outputLayer});
            NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(train, network, measure);
            
            OptimizationAlgorithm trainer = new RandomizedHillClimbing(nnop);
            
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10, 9);

            Instance optimalInstance = trainer.getOptimal();
            network.setWeights(optimalInstance.getData());

            double predicted, actual;
            double trainError = 0;
            double testError = 0;
            Instance[] trainInstances = train.getInstances();
            Instance[] testInstances = test.getInstances();
            
            for(int j = 0; j < trainInstances.length; j++) {
                network.setInputValues(trainInstances[j].getData());
                network.run();

                Instance output = trainInstances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                trainError += measure.value(output, example) / trainInstances.length;
            }
            
            trainErrors.add(trainError);
            
            for (int j = 0; j < testInstances.length; j++) {
                network.setInputValues(testInstances[j].getData());
                network.run();
                
                Instance output = testInstances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                testError += measure.value(output, example) / testInstances.length;
            }
            
            testErrors.add(testError);
            
            System.out.println("Train Erro : " + trainError + " Test Error: " + testError);
        }
                
        List<String> output_lines = new ArrayList<>();
        for (int i = 0; i < trainErrors.size(); i++) {
            String s = i + "," + trainErrors.get(i) + "," + testErrors.get(i);
            output_lines.add(s);
        }

        try {
            Path file = Paths.get("src/opt/test/Adult_RHC_Restarts_Results.csv");
            Files.write(file, output_lines, Charset.forName("UTF-8"));
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    private static Instance[] initializeInstances() {
        
        double[][][] attributes = new double[32561][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/test/adult.csv")));
            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[4]; // 4 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < 4; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            System.out.println("crashed here");
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            instances[i].setLabel(new Instance(attributes[i][1][0]));
        }

        return instances;
    }
}
