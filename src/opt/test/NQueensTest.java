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
import opt.ga.NQueensFitnessFunction;
import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.SwapNeighbor;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.SwapMutation;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * @author kmanda1
 * @version 1.0
 */
public class NQueensTest {
    /** The n value */
    private static final int N = 25;
    /** The t value */

    private static List<Double> rhcList = new ArrayList<>();
    private static List<Long> rhcTimes = new ArrayList<>();
    private static List<Double> saList = new ArrayList<>();
    private static List<Long> saTimes = new ArrayList<>();
    private static List<Double> gaList = new ArrayList<>();
    private static List<Long> gaTimes = new ArrayList<>();
    private static List<Double> mimicList = new ArrayList<>();
    private static List<Long> mimicTimes = new ArrayList<>();
    private static List<String> lines = new ArrayList<>();
    
    public static void main(String[] args) {
        int[] ranges = new int[N];
        Random random = new Random(N);
        for (int i = 0; i < N; i++) {
        	ranges[i] = random.nextInt();
        }
        NQueensFitnessFunction ef = new NQueensFitnessFunction();
        Distribution odd = new DiscretePermutationDistribution(N);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);   
        SimulatedAnnealing sa = new SimulatedAnnealing(1E3, .1, hcp);
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 10, 10, gap);
        MIMIC mimic = new MIMIC(200, 10, pop);
        
        lines.add("Iteration,RHC,SA,GA,MIMIC");
            
        for(int i = 0; i < 100; i++)
        {
            Double rhcVal = rhc.train();
            Double saVal = sa.train();
            Double gaVal = ga.train();
            Double mimicVal = mimic.train();
            
            lines.add(i + ", " + rhcVal + "," + saVal + "," + gaVal + "," + mimicVal);
        }

        try {
            Path file = Paths.get("src/opt/test/NQueens.csv");
            Files.write(file, lines, Charset.forName("UTF-8"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static int lowestMax(List<Double> dubs) {
        double max = dubs.get(dubs.size() - 1);
        for (int i = dubs.size() - 1; i >= 0; i--) {
            if (dubs.get(i) < max) {
                return i + 1;
            }
        }

        return -1;
    }
}
