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
    private static final int N = 20;
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
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 100);
        fit.train(rhcList, rhcTimes);
        long starttime = System.currentTimeMillis();
        System.out.println(ef.value(rhc.getOptimal()) + " " + lowestMax(rhcList) + " " + String.valueOf(rhcTimes.get(lowestMax(rhcList)) - rhcTimes.get(0)));
        System.out.println("RHC: Board Position: ");
        System.out.println(ef.boardPositions());
        System.out.println("Time : "+ (System.currentTimeMillis() - starttime));
        
        System.out.println("============================");
        
        SimulatedAnnealing sa = new SimulatedAnnealing(1E1, .1, hcp);
        fit = new FixedIterationTrainer(sa, 100);
        fit.train(saList, saTimes);
        
        starttime = System.currentTimeMillis();
        System.out.println(ef.value(sa.getOptimal()) + " " + lowestMax(saList) + " " + String.valueOf(saTimes.get(lowestMax(saList)) - saTimes.get(0)));
        System.out.println("SA: Board Position: ");
        System.out.println(ef.boardPositions());
        System.out.println("Time : "+ (System.currentTimeMillis() - starttime));
        
        System.out.println("============================");
        
        starttime = System.currentTimeMillis();
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 0, 10, gap);
        fit = new FixedIterationTrainer(ga, 100);
        fit.train(gaList, gaTimes);
        System.out.println(ef.value(ga.getOptimal()) + " " + lowestMax(gaList) + " " + String.valueOf(gaTimes.get(lowestMax(gaList)) - gaTimes.get(0)));
        System.out.println("GA: Board Position: ");
        System.out.println(ef.boardPositions());
        System.out.println("Time : "+ (System.currentTimeMillis() - starttime));
        
        System.out.println("============================");
        
        starttime = System.currentTimeMillis();
        MIMIC mimic = new MIMIC(200, 10, pop);
        fit = new FixedIterationTrainer(mimic, 100);
        fit.train(mimicList, mimicTimes);
        System.out.println(ef.value(mimic.getOptimal()) + " " + lowestMax(mimicList) + " " + String.valueOf(mimicTimes.get(lowestMax(mimicList)) - mimicTimes.get(0)));
        System.out.println("MIMIC: Board Position: ");
        System.out.println(ef.boardPositions());
        System.out.println("Time : "+ (System.currentTimeMillis() - starttime));

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
