import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;

/**
 * Evaluates classification confidence scores and the area under the receiver operating characteristic's effectivenes.
 * 
 * 
 * java -cp bin Evaluation spam ../datasets/sms/sms_predictions/dev_knn_k10_spam_cosine.csv
 * 
 * java -cp bin Evaluation spam ../datasets/sms/sms_predictions/dev_knn_k10_spam_euclidean.csv
 * 
 * java -cp bin Evaluation spam ../datasets/sms/sms_predictions/dev_knn_k10_spam_manhattan.csv
 * 
 * 
 * 
 * @author Eva Rubio.
 * (Built off the codebase provided by: Professor Hank Feild)
 */

/**
 * Some physical external resources I used are:
 * -- Data Mining Concepts and Techniques. 
 *          Pages 373-377.
 *          Authors: Jiawei Han, Micheline Kamber & Jian Pei.
 * -- Artificial Intelligence: A Modern Approach. 
 *          Pages: A bit of everywhere..
 *          Authors: Stuart Russell & Peter Norvig.
 * 
 * Online external resources used to better understand the assignment as a whole: 
 * 
 * -- https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5 
 *          (Best one I found)
 * -- https://www.youtube.com/watch?v=AqCIhh88Lxw&feature=youtu.be
 *          This video was awesome. What I found extremely helpful was to re-watch it whenever I was stuck on this part!
 * 
 */

public class Evaluation {

    public ArrayList<String> trueLabels;
    public ArrayList<Double> predictedLabels;
    public HashSet<String> distinctLabels;
    public String positiveClass;
    //public double threshold;

    public void setPositiveClass(String givenPosClass) {
        this.positiveClass = givenPosClass;
    }

    public String getPosClass() {
        return this.positiveClass;
    }


    /**
     * A container for evaluation statistics.
     */
    public class EvaluationStats {
        public double tp, fp, tn, fn, accuracy, precision, recall, f1, fprate;


        /**
         * Initializes all the stats to 0.
         * 
         * @param positiveClass The label to consider the positive classification in the
         *                      confusion matrix. All others are considered negative
         *                      classifications.
         */
        public EvaluationStats() {
            tp = 0;
            fn = 0;
            tn = 0;
            fp = 0;
            accuracy = 0;
            precision = 0;
            recall = 0;
            f1 = 0;
            fprate = 0;

            //this.positiveClass = positiveClass;
        }

        /**
         * Pretty prints the stats to stdout.
         
        public void display() {
            System.out.println("Positive class:  " + getPosClass());
            System.out.printf("True positives:  %5.0f\n", tp);
            System.out.printf("True negatives:  %5.0f\n", tn);
            System.out.printf("False positives: %5.0f\n", fp);
            System.out.printf("False negatives: %5.0f\n", fn);
            System.out.println("---------");
            System.out.printf("Accuracy:        %5.3f\n", accuracy);
            System.out.printf("Precision:       %5.3f\n", precision);
            System.out.printf("Recall:          %5.3f\n", recall);
            System.out.printf("F1:              %5.3f\n", f1);
        }
*/
        /**
         * Derives advanced measures (e.g., accuracy) from the confusion matrix stats.
         * Measures for how well our model did. 
         */
        public void computeStatsFromConfusionMatrix() {
            // How many things were labeled correctly. Treats all clasess the same. 
            accuracy = (tp + tn) / (tp + tn + fp + fn);
            //Fraction of observations classified as the positive class that were trully the possivie class. Favors the pos class. 
            precision = tp / (tp + fp);
            //Of all the obersvations with true labels belonging to the positive class, how many did we capture. 
            recall = tp / (tp + fn);
            //combines both precision and recall. (harmonic mean).
            f1 = 2 * tp / (2 * tp + fp + fn);
            // the fraction of the negative ones missclasified as the possitive class. 
            fprate = fp / (tn + fp);
        }

        
    }

    /**
     * Extracts the true and predicted labels from the given file and the set of
     * distinct true labels. These are stored in the Evaluation class members.
     * 
     * @param filename The name of a CSV file, with a header, where the last two
     *                 columns are true label and predicted label (in that order).
     */
    public void parseInputFile(String filename) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        trueLabels = new ArrayList<String>();
        predictedLabels = new ArrayList<Double>();
        distinctLabels = new HashSet<String>();

        reader.readLine();
        while (reader.ready()) {
            String[] columns = reader.readLine().split(",");
            if (columns.length < 2)
                continue;

            trueLabels.add(columns[columns.length - 2]);
            predictedLabels.add(Double.parseDouble(columns[columns.length - 1]));

            distinctLabels.add(columns[columns.length - 2]);
        }
        reader.close();
        //organizeData();
        /* for (int i = 0; i < trueLabels.size(); i++) {
            System.out.println(trueLabels.get(i) + ", " + predictedLabels.get(i));
        } */
    }

    /**
     * Computes the evaluation statistics over the true and predicted labels given a
     * label to use as the positive classification.
     * 
     * @param positiveClass The label to consider the positive classification in the
     *                      confusion matrix. All others are considered negative
     *                      classifications.
     * @return The evaluation stats. 
     */
    public EvaluationStats evaluate( double thresh) {
        EvaluationStats stats = new EvaluationStats();
        //threshold = 0.5;
        //System.out.println(threshold);

        for (int i = 0; i < trueLabels.size(); i++) {
            // if (trueLabels.get(i).equals(predictedLabels.get(i)))
            //if true and predicted match:
            if (thresh <= predictedLabels.get(i))
                if (trueLabels.get(i).equals(getPosClass()))
                    stats.tp++;
                else
                    stats.fp++;
            else if (trueLabels.get(i).equals(getPosClass()))
                stats.fn++;
            else
                stats.tn++;
        }

        stats.computeStatsFromConfusionMatrix();
        return stats;
    }


    public class Obser {
        //double threshold;
        String trueLab;
        double score;
        

        public Obser(String trueLab, double score) {
            //this.threshold = threshold;
           //threshold = 0.0;
            this.trueLab = trueLab;
            this.score = score;

        }

        public double getScore() {
            return this.score;
        }

        public String getTrueLab() {
            return this.trueLab;
        }

    }

    /**
     * any confidence score in predictedlabel that is = or > than the threshold : will be labaled as the positive class.
     * 
     * creates list of observations & sorts them from smaller to larger, based on their confidence score. 
     */
    public ArrayList<Obser> organizeData() {
        // 
        ArrayList<Obser> obserList = new ArrayList<Obser>();

        for (int i = 0; i < trueLabels.size(); i++) {
            obserList.add(new Obser(trueLabels.get(i), predictedLabels.get(i)));
            //System.out.print(predictedLabels.get(i));
        }

        //sorts the list for future use.
        // https://stackoverflow.com/questions/2784514/sort-arraylist-of-custom-objects-by-property
        Collections.sort(obserList, new Comparator<Obser>() {
            public int compare(Obser obs1, Obser obs2) {
                return Double.compare(obs1.getScore(), obs2.getScore());
            }
        });

        /* for (Obser ob : obserList) {
            System.out.println(ob.getTrueLab() + ", " + ob.getScore());
        } */

        return obserList;

    }

    // https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5
    public class RocCurve {
        double threshold;
        double falsePosRate;
        double recallRate;

        public RocCurve(double threshold, double falsePosRate, double recallRate) {
            this.threshold = threshold;
            this.falsePosRate = falsePosRate;
            this.recallRate = recallRate;
        }
    }

    /**
     * Identifies ALL the thresholds in our model.
     * 
     * The Starting threshold is = the score of the lowest scoring observation.
     * 
     * Each threshold has associated to it some recall and false positive rate.
     * 
     * ensures that you are only calculating recall and the false positive rate at
     * thresholds where those values will change.
     * the ROC curve gives us a way to evaluate the model over ALL possible thresholds. 
     */
    public ArrayList<RocCurve> curveRocCalculator(ArrayList<Obser> obserList) {
        double currentThresh = 0.0;
        double lastScoreSeen = 0.0;
        ArrayList<RocCurve> curvePoints = null;
        EvaluationStats stats = new EvaluationStats();
        curvePoints = new ArrayList<RocCurve>();

        // The Starting threshold is = the score of the lowest scoring observation.
        //urrentThresh = obserList.get(0).getScore();

        // calculate recall and falsePos for this threshold.
        stats = evaluate(currentThresh);
        curvePoints.add(new RocCurve(currentThresh, stats.fprate, stats.recall));

        for (Obser obs : obserList) {
            currentThresh = obs.getScore();
           
            // if the thresh is different to the one we currently have 
            if (lastScoreSeen != currentThresh) {

                // calculate recall and falsePos for this threshold.
                stats = evaluate(currentThresh);
                curvePoints.add(new RocCurve(currentThresh, stats.fprate, stats.recall));
            }
            lastScoreSeen = obs.getScore();
        }
        // sorts the list for future use.
        // https://stackoverflow.com/questions/2784514/sort-arraylist-of-custom-objects-by-property
        Collections.sort(curvePoints, new Comparator<RocCurve>() {
            public int compare(RocCurve point1, RocCurve point2) {
                return Double.compare(point1.falsePosRate, point2.falsePosRate);
            }
        });

        return curvePoints;
    }

    /***
     * Calculates the area under the Receiver Operating Curve. i.e. the AUROC.
     * Calculates the area of each rectangle in the graph and sums all the areas together.
     * 
     * The output represents the probability that a classification model scores a randomly chosen
     * positive observation HIGHER, than a randomly chosen negative observation. 
     * 
     * Any result above 0.5 is considered to be good, and 
     * any result closer to 1.0 is great.
     * 
     * @param curvePoints An ArrayList representing the points that form the ROC curve.
     * @return the total area under the given ROC curve.
     */
    public double areaUC(ArrayList<RocCurve> curvePoints) {

        double area = 0.0;

        
        /*         for (int i = 0; i < curvePoints.size(); i++) {
            currentPair = i;
        
            // first rectangle
            if (i == 0) {
                widthX = curvePoints.get(i).falsePosRate;
                //hightY = curvePoints.get(i).recallRate;
            } else {
                // ex.  ( x3 - x2 )
                widthX = curvePoints.get(i).falsePosRate - curvePoints.get(i - 1).falsePosRate;
            }
        
            hightY = curvePoints.get(i).recallRate;
            area += widthX * hightY; */
            
        for (int i = 1; i < curvePoints.size(); i++) {
            
            area += (curvePoints.get(i).falsePosRate - curvePoints.get(i - 1).falsePosRate)
                    * curvePoints.get(i).recallRate;

        }
        area += (1 - curvePoints.get(curvePoints.size()-1).falsePosRate) * curvePoints.get(
                curvePoints.size() - 1).recallRate;

        
        return area;
    }

    /**
     * -----------------------------------------------------------------------------------------------------------------
     * Reads in a filename from the command line arguments and computes the
     * evaluation stats. Stats are displayed to stdout.
     * 
     * @param args The first argument should be the file to evaluate.
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {
        Evaluation evaluation = new Evaluation();
        // this is the output of our kNNPA4.java classification program.
        String inputFilename;
        String givenPositiveClass;
        ArrayList<Obser> obsList = new ArrayList<Obser>();

        ArrayList<RocCurve> curvePoints = new ArrayList<RocCurve>();

        double areaUC;

        if (args.length < 2) {
            System.err.println("Usage: Evaluation <positive class> <file>\n"
                    +
             "\n<file> can have any number of comma separated "
                    + "columns; the final two should be\nthe true label and the "
                    + "predicted label (in that order). It should have a header.\n"+
                "<positive class> must : \n"+
                "   * be a string\n"+
                "   * match exaclty 1 of the classes in the dataset.");
            System.exit(1);
        }

        givenPositiveClass = args[0];
        inputFilename = args[1];

        
        evaluation.setPositiveClass(givenPositiveClass);
        evaluation.parseInputFile(inputFilename);
        obsList = evaluation.organizeData();



        curvePoints = evaluation.curveRocCalculator(obsList);

       
     
        
        System.out.println();
        System.out.println("====================================");

        System.out.println("Positive Class: " + evaluation.getPosClass());

       

        areaUC = evaluation.areaUC(curvePoints);
        System.out.println("Area Under Curve: " + areaUC);
        System.out.println("====================================");



        
    }
}