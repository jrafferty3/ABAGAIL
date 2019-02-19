package opt.test;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * A test of the knapsack problem
 *
 * Given a set of items, each with a weight and a value, determine the number of each item to include in a
 * collection so that the total weight is less than or equal to a given limit and the total value is as
 * large as possible.
 * https://en.wikipedia.org/wiki/Knapsack_problem
 *
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class KnapsackTest {
    /** Random number generator */
    private static final Random random = new Random();
    /** The number of items */
    private static final int NUM_ITEMS = 40;
    /** The number of copies each */
    private static final int COPIES_EACH = 4;
    /** The maximum value for a single element */
    private static final double MAX_VALUE = 50;
    /** The maximum weight for a single element */
    private static final double MAX_WEIGHT = 50;
    /** The maximum weight for the knapsack */
    private static final double MAX_KNAPSACK_WEIGHT =
         MAX_WEIGHT * NUM_ITEMS * COPIES_EACH * .4;
         
    private static List<Double> rhcList = new ArrayList<>();
    private static List<Long> rhcTimes = new ArrayList<>();
    private static List<Double> saList = new ArrayList<>();
    private static List<Long> saTimes = new ArrayList<>();
    private static List<Double> gaList = new ArrayList<>();
    private static List<Long> gaTimes = new ArrayList<>();
    private static List<Double> mimicList = new ArrayList<>();
    private static List<Long> mimicTimes = new ArrayList<>();
    private static List<String> lines = new ArrayList<>();

    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        int[] copies = new int[NUM_ITEMS];
        Arrays.fill(copies, COPIES_EACH);
        double[] values = new double[NUM_ITEMS];
        double[] weights = new double[NUM_ITEMS];
        for (int i = 0; i < NUM_ITEMS; i++) {
            values[i] = random.nextDouble() * MAX_VALUE;
            weights[i] = random.nextDouble() * MAX_WEIGHT;
        }
        int[] ranges = new int[NUM_ITEMS];
        Arrays.fill(ranges, COPIES_EACH + 1);

        EvaluationFunction ef = new KnapsackEvaluationFunction(values, weights, MAX_KNAPSACK_WEIGHT, copies);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);

        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges);

        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 1000);
        fit.train(rhcList, rhcTimes);
        System.out.println(ef.value(rhc.getOptimal()));
        
        SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
        fit = new FixedIterationTrainer(sa, 1000);
        fit.train(saList, saTimes);
        System.out.println(ef.value(sa.getOptimal()));
        
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(300, 150, 50, gap);
        fit = new FixedIterationTrainer(ga, 1000);
        fit.train(gaList, gaTimes);
        System.out.println(ef.value(ga.getOptimal()));
        
        MIMIC mimic = new MIMIC(200, 100, pop);
        fit = new FixedIterationTrainer(mimic, 1000);
        fit.train(mimicList, mimicTimes);
        System.out.println(ef.value(mimic.getOptimal()));
        
        
        for (int i = 0; i < rhcList.size(); i++) {
            String rhcVal = (i < rhcList.size()) ? String.valueOf(rhcList.get(i)) + ", " : String.valueOf(rhcList.get(rhcList.size() - 1));
            String saVal = (i < saList.size()) ? String.valueOf(saList.get(i)) + ", " : String.valueOf(saList.get(saList.size() - 1));
            String gaVal = (i < gaList.size()) ? String.valueOf(gaList.get(i)) + ", " : String.valueOf(gaList.get(gaList.size() - 1));
            String mimicVal = (i < mimicList.size()) ? String.valueOf(mimicList.get(i)) + ", " : String.valueOf(mimicList.get(mimicList.size() - 1));
            String rhcTime = (i < rhcTimes.size()) ? String.valueOf(rhcTimes.get(i)) + ", " : String.valueOf(rhcTimes.get(rhcTimes.size() - 1));
            String saTime = (i < saTimes.size()) ? String.valueOf(saTimes.get(i)) + ", " : String.valueOf(saTimes.get(saTimes.size() - 1));
            String gaTime = (i < gaTimes.size()) ? String.valueOf(gaTimes.get(i)) + ", " : String.valueOf(gaTimes.get(gaTimes.size() - 1));
            String mimicTime = (i < mimicTimes.size()) ? String.valueOf(mimicTimes.get(i)) + ", " : String.valueOf(mimicTimes.get(mimicTimes.size() - 1));

            lines.add(i + ", " + rhcVal + saVal + gaVal + mimicVal + rhcTime + saTime + gaTime + mimicTime);
        }

        try {
            Path file = Paths.get("src/opt/test/Knapsack.csv");
            Files.write(file, lines, Charset.forName("UTF-8"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
