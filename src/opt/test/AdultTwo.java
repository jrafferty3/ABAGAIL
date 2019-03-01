package opt.test;

import func.nn.feedfwd.FeedForwardNetwork;
import func.nn.feedfwd.FeedForwardNeuralNetworkFactory;
import opt.OptimizationAlgorithm;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import opt.SimulatedAnnealing;
import opt.RandomizedHillClimbing;
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

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer 
 * or more than 15 rings. 
 *
 * @author Hannah Lau
 * @version 1.0
 */
public class AdultTwo {
    private static Instance[] instances = initializeInstances();

    private static int inputLayer = 4, outputLayer = 1, trainingIterations = 1000;
    private static FeedForwardNeuralNetworkFactory factory = new FeedForwardNeuralNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    //private static FeedForwardNetwork networks[] = new FeedForwardNetwork[100];
    private static FeedForwardNetwork networks[] = new FeedForwardNetwork[2];
    //private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[100];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[2];

    //private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[100];
    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[2];
    //private static String[] oaNames = new String[100];
    //private static String[] oaNames = new String[3];
    private static String[] oaNames = {"RHC", "SA"};
    private static String results = "";
    private static List<List<Double>> oaResultsTrain = new ArrayList<>();
    private static List<List<Double>> oaResultsTest = new ArrayList<>();

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        new RandomOrderFilter().filter(set);
        TestTrainSplitFilter ttsf = new TestTrainSplitFilter(70);
        ttsf.filter(set);
        DataSet train = ttsf.getTrainingSet();
        DataSet test = ttsf.getTestingSet();
        
        for(int k = 0; k < 2; k++){
            
            for (int i = 0; i < trainingIterations; i++) {
                oaResultsTrain.add(new ArrayList<>());
                oaResultsTest.add(new ArrayList<>());
            }
            
            for(int i = 0; i < oa.length; i++) {
                networks[i] = factory.createClassificationNetwork(
                        new int[] {inputLayer, 14, 14, 14, 14, outputLayer});
                nnop[i] = new NeuralNetworkOptimizationProblem(train, networks[i], measure);
            }
            
            oa[0] = new RandomizedHillClimbing(nnop[0]);
            oa[1] = new SimulatedAnnealing(1E8, .65, nnop[1]);
            //oa[2] = new StandardGeneticAlgorithm(50, 35, 75, nnop[2]);
            
            for (int i = 0; i < oa.length; i++) {
                double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
                train(oa[i], networks[i], oaNames[i], train, test); //trainer.train();
                end = System.nanoTime();
                trainingTime = end - start;
                trainingTime /= Math.pow(10, 9);
            }
            
        }
        
        List<String> output_lines = new ArrayList<>();
        output_lines.add("Iteration,RHC Train,SA Train,RHC Test,SA Test");
        for (int i = 0; i <trainingIterations; i++) {
            String s = (i + 1) + ",";
            for(int j = 0; j < oa.length; j++){
                s = s.concat(oaResultsTrain.get(i).get(j) + ",");
            }
            for(int j = 0; j < oa.length; j++){
                if( i < oaResultsTest.size()){
                    s = s.concat(oaResultsTest.get(i).get(j) + ",");
                }
            }
            output_lines.add(s);
        }

        try {
            Path file = Paths.get("src/opt/test/Adult_Two_Results.csv");
            Files.write(file, output_lines, Charset.forName("UTF-8"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void train(OptimizationAlgorithm oa, FeedForwardNetwork network, String oaName, DataSet train, DataSet test) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");
        Instance[] trainInstances = train.getInstances();
        Instance[] testInstances = test.getInstances();

        //double lastError = 0;
        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double trainError = 0;
            double testError = 0;
            for(int j = 0; j < trainInstances.length; j++) {
                network.setInputValues(trainInstances[j].getData());
                network.run();

                Instance output = trainInstances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                trainError += measure.value(output, example) / trainInstances.length;
            }
            
            for (int j = 0; j < testInstances.length; j++) {
                network.setInputValues(testInstances[j].getData());
                network.run();
                
                Instance output = testInstances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                testError += measure.value(output, example) / testInstances.length;
            }

            System.out.println("Iteration " + String.format("%04d" ,i) + ": " + df.format(trainError) + " " + df.format(testError));
            oaResultsTrain.get(i).add(trainError);
            oaResultsTest.get(i).add(testError);
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
