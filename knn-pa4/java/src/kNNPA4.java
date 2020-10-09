import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * cd Desktop/ENDI-2020/ARTIF-INTEL/PA4-KNN-AUC/knn-pa4/java
 * 
 * javac -d bin src/*.java
 * 
 * 
 * java -cp bin kNNPA4 ../datasets/sms/train.csv ../datasets/sms/dev.csv 10 cosine spam > ../datasets/sms/sms_predictions/dev_knn_k10_spam_cosine.csv
 * 
 * java -cp bin kNNPA4 ../datasets/sms/train.csv ../datasets/sms/dev.csv 10 euclidean spam > ../datasets/sms/sms_predictions/dev_knn_k10_spam_euclidean.csv
 * 
 * java -cp bin kNNPA4 ../datasets/sms/train.csv ../datasets/sms/dev.csv 10 manhattan spam > ../datasets/sms/sms_predictions/dev_knn_k10_spam_manhattan.csv
 * 
 * Modified version of the k-Nearest Neighbors Classification Algorithm.
 *      Instead of predicting the class label of an observation, this program outputs the Confidence Score 
 *          that an observation is from the provided Positive Class. 
 *          i.e. Instead of outputting the majority label of an observation, it outputs the percentage 
 *          of an observation's <k> Nearest Neighbors that belong to the given positive class. 
 * 
 * @author Eva Rubio. 
 * (Built off the codebase provided by: Professor Hank Feild)
 */

public class kNNPA4 {
    
    public class Observation {
        public ArrayList<Double> features;
        public String label;

        public Observation(ArrayList<Double> features, String label) {
            this.features = features;
            this.label = label;
        }

        public String toString() {
            StringBuffer output = new StringBuffer();
            String lastCol;

            for(int i = 0; i < features.size(); i++){
                output.append(features.get(i));
                if (i < features.size() - 1) {
                    output.append(",");
                }
                if (i == features.size()) {
                    lastCol = String.format("%.5f", features.get(i));
                    output.append(lastCol);
                }
            }
            if(label != null)
                output.append(",").append(label);

            return output.toString();
        }
    }

    public class Dataset {
        public ArrayList<String> columnNames;
        public ArrayList<Observation> observations;

        public Dataset(ArrayList<String> columnNames, ArrayList<Observation> observations) {
            this.columnNames = columnNames;
            this.observations = observations;
        }

        public String columnNamesAsCSV() {
            StringBuffer output = new StringBuffer();

            for (int i = 0; i < columnNames.size(); i++) {
                output.append(columnNames.get(i));
                if (i < columnNames.size() - 1)
                    output.append(",");
            }

            return output.toString();
        }
    }
    
    /**
     * Searching for NaN.
     * 
     * (1) mathematically undefined numerical operations: - ZERO / ZERO = NaN -
     * INFINITY - INFINITY = NaN - INFINITY * ZERO = NaN
     * 
     * (2) Numerical operations which don’t have results in real numbers: - SQUARE
     * ROOT OF -1 = NaN - LOG OF -1 = NaN
     * 
     * (3) All numeric operations with NaN as an operand produce NaN as a result: - 2
     * + NaN = NaN - 2 - NaN = NaN - 2 * NaN = NaN - 2 / NaN = NaN
     * 
     * // 1. static method if (Double.isNaN(doubleValue)) { ... } // 2. object's
     * method if (doubleObject.isNaN()) { ... }
     * 
     * 
     * - if 0/0 -- invalid operation - if .sqrt(-1) -- unrepresentable values
     * 
     * “A constant holding a Not-a-Number (NaN) value of type double. It is
     * equivalent to the value returned by
     * Double. longBitsToDouble(0x7ff8000000000000L)
     * https://www.baeldung.com/java-not-a-number
     * 
     * Calculates the Euclidean distance between 2 observations. 
     */

    public class EuclideanDistance implements Distance {
        public double distance(ArrayList<Double> featuresA, ArrayList<Double> featuresB) {
            double sum = 0;
            for (int i = 0; i < featuresA.size(); i++)
                sum += Math.pow(featuresA.get(i) - featuresB.get(i), 2);
            return Math.sqrt(sum);
        }
    }

    /**
     * Most of these are just theory... It took me a looong time to remember what trigonomtry was...
     * 
     * https://www.machinelearningplus.com/nlp/cosine-similarity/
     * https://www.khanacademy.org/math/precalculus/x9e81a4f98389efdf:vectors/x9e81a4f98389efdf:component-form/a/vector-magnitude-and-direction-review
     * https://www.machinelearningplus.com/nlp/cosine-similarity/
     * https://www.lasclasesdegonzalo.com/trigonometricosparaangulosconcretos
     * https://www.geogebra.org/m/tezrgs9Z
     * 
     * Calculates the Euclidean distance between 2 observations. 
     * 
     * For 2 Observation feature vectors A and B.
     * 
     * - Similarity funct: - Range [-1, 1]. - Highier = Closer. This is what we want!!
     * 
     * - DISTANCE fuct (): - Range [0, 1]. - Lower = Closer.
     * 
     * For us to compute de DIST(), we first need to calculate the COSINE-SIMILARITY()
     */
    public class CosineSimilarity implements Distance {
        /**
         * Calculates the cosine similarity between 2 observations. 
         * Numerator:
         * For each feature, multiply the value of that feature in obsA with the value of that feature in obsB.
         *      We add those products together to get our numerator. 
         * 
         * Denominator:
         *  Squares each feature value in obsA, adds them together
         *  and, takes the square root .
         * Does the same for obsB.
         * Multiplies the 2 square roots together. 
         * 
         * @param featuresA
         * @param featuresB
         * @return The similarity between the 2 observations. 
         */
        public double cosineSimilarity(ArrayList<Double> featuresA, ArrayList<Double> featuresB) {
            double product = 0.0;
            double denom = 0.0;
            double sqrA = 0.0;
            double sqrB = 0.0;
            double sol = 0.0;

            for (int i = 0; i < featuresA.size(); i++) {

                product += featuresA.get(i) * featuresB.get(i);

                sqrA += Math.pow(featuresA.get(i), 2);

                sqrB += Math.pow(featuresB.get(i), 2);
            }
            //if (sqrA == 0 && sqrB == 0) { return 2.0; }
            denom = Math.sqrt(sqrA) * Math.sqrt(sqrB);

            sol = product / denom;

            return sol;
        }

        // Calculates the distance between 2 observations using their cosine similarity.
        public double distance(ArrayList<Double> featuresA, ArrayList<Double> featuresB) {
            double thecos = cosineSimilarity(featuresA, featuresB);
            double numer = -1 * thecos + 1;
            return numer / 2; 
        }
    }


    
    public class ManhattanDistance implements Distance {
        public double distance(ArrayList<Double> featuresA, ArrayList<Double> featuresB){
            double sum = 0;
            for(int i = 0; i < featuresA.size(); i++)
                sum += Math.abs(featuresA.get(i)-featuresB.get(i));
            return sum;
        }
    }

    /**
     * Generate a list of observations from the data file.
     * 
     * @param filename The name of the file to parse. Should have a header and be in
     *                 comma separated value (CSV) format.
     * @param hasLabel If true, the last column will be used as the label for each
     *                 observation and all other columns will be features. If false,
     *                 *all* columns will be used as features.
     * @return The observations and header (column names).
     * @throws IOException
     */
    public Dataset parseDataFile(String filename, boolean hasLabel) throws IOException {
        ArrayList<String> columnNames = new ArrayList<String>();
        ArrayList<Observation> observations = new ArrayList<Observation>();
        BufferedReader reader = new BufferedReader(new FileReader(filename));

        for(String col : reader.readLine().split(",")){
            columnNames.add(col);
        }

        while(reader.ready()){
            String[] columns = reader.readLine().split(",");
            ArrayList<Double> features = new ArrayList<Double>();
            String label = null;
            if(hasLabel){
                for(int i = 0; i < columns.length-1; i++){
                    features.add(Double.parseDouble(columns[i]));
                }
                label = columns[columns.length-1];
            } else {
                for(int i = 0; i < columns.length; i++){
                    features.add(Double.parseDouble(columns[i]));
                }
            }
            
            observations.add(new Observation(features, label));
        }
        reader.close();

        return new Dataset(columnNames, observations);
    }

    /**
     * Predict the label of newObservation based on the majority label among 
     * the k nearest observations in trainingSet.
     * 
     * @param newObservation The observation to classify.
     * @param trainingSet The set of observations to derive the label from.
     * @param k The number of nearest neighbors to consider when determining the label.
     * @param dist The distance function to use.
     * @return The predicted label of newObservation.
     */
    public Double classify(Observation newObs, 
            ArrayList<Observation> trainingSet, int k, Distance dist, String positiveClass) {

                // the list of the distance from newObs to each of its k-neighbors. 
        ArrayList<Double> distancesList = new ArrayList<Double>();
        ArrayList<String> labelsList = new ArrayList<String>();

        HashMap<String, Integer> labelsDistSum = new HashMap<String, Integer>();

        int maxDistIndex; // the position of the farther obs in the list of distances. 
        double maxDist; // is the largest distance that one of our neighbors has. I.e. The obs which is the farthest from newObs.

        int totalCounts = 0;
        int posCounts = 0;
       // double sol = 0;

        // we calculate the distance from newObs to EVERY single obs in the given dataset, 
        // and store k of them in 'distancesList'.
        for (Observation observation : trainingSet) {

            // for this 'observation'
            Double distance = dist.distance(observation.features, newObs.features);
            // if the list doesnt contain k neighbors in it:
            if (distancesList.size() < k) {
                // add the distance to this neighbor 'observation', to the list
                distancesList.add(distance);
                // and also add its corresponding label
                labelsList.add(observation.label);

            } else { // if we already have the list with  k neighbors:

                // Find the largest value in distancesList and it's corresponding index.

                maxDistIndex = 0;
                maxDist = distancesList.get(0); // we set it to the fisrt dist val as it is so far, the largest one seen.

                // we start looping at 1 because we have already seen 0.
                for (int i = 1; i < distancesList.size(); i++) {

                    if (distancesList.get(i) > maxDist) {
                        maxDist = distancesList.get(i);
                        maxDistIndex = i;
                    }
                }

                // Replace the largest distance with this one if this one is
                // smaller.
                // if the distance we just computed (between observation and newObs)
                // is smaller than the largest distance that our list contains so far, replace it.
                // Remember! We want the closest neighbors!
                if (distance < maxDist) {
                    // .set(index, newVal)  --> Replace the current value at 'index', with: 'newVal'.
                    // replace our current largest value (located at maxDistIndex) with this smaller distance value.
                    distancesList.set(maxDistIndex, distance);
                    // we find the item located at the same index, and we replace the label that was there, with this one.
                    labelsList.set(maxDistIndex, observation.label);
                }
            }
        }
        // -------------------------------------------------------------------------------------------------------------------------

        //Now that we have located our k nearest neighbors and their corresponig labels:

        //add all the distance values of the observations with posClass.
        // add all the distance values of the observations with negativeClass.
        //add them together to get the total distance.
        

        // Add labels to the hash map.
        for (String label : labelsList) {
            // if the label has already been added as a key to the map:
            if (labelsDistSum.containsKey(label)) {
                // replace the old count for this label with oldCount+1.
                labelsDistSum.put(label, labelsDistSum.get(label) + 1);
            } else {
                // if the label has neever been seen before, add it to the map and set the count
                // to be 1.
                labelsDistSum.put(label, 1);
            }
        }
        for (String label : labelsDistSum.keySet()) {

            totalCounts += labelsDistSum.get(label);

            if (labelsDistSum.containsKey(positiveClass)) {
                posCounts = labelsDistSum.get(positiveClass);
            } else {
                //If the class is not present, the confidence socre is null. 
                return 0.0;
            }
        }
        Double totalAsDouble = Double.valueOf(totalCounts);
        Double posCountsAsDouble = Double.valueOf(posCounts);


        return posCountsAsDouble / totalAsDouble;
    }

    /**
     * https://pdfs.semanticscholar.org/0318/f6c2c8928d3259cd8d65283901536cea9e33.pdf?_ga=2.140176563.1629213328.1588025057-1310319897.1581406461
     * 
     * Predict the Confidence Score of newObservation belonging to the positive
     * class based on the percentage of its k nearest observations in trainingSet.
     * It outputs a Confidence Score that this observation is from the given
     * positiveClass. The score computed, inticates the confidence that an
     * observation belongs to the positive class.
     * 
     * - if score is: - closer to 1 It means that, newObs is more likely to be from
     * the positiveClass. - closer to 0 Means it is more likely for newObs to be
     * from the negativeClass.
     * 
     * ---- the confidence score (in terms of this distance measure) is the relative
     * distance.
     * 
     * For example, if sample S1 has a distance 80 to Class 1 and distance 120 to
     * Class 2,
     * 
     * - then it has (100-(80/200))%=60% confidence to be in Class 1 - and 40%
     * confidence to be in Class 2.
     * 
     * find its k nearest neighbors. 
     * find what their labels are.  
     * get the percentage of those neighbors that have the label=positiveClass.
     * 
     * @param newObservation The observation to get the Confidence Score from.
     * @param trainingSet    The set of observations to derive the score from.
     * @param k              The number of nearest neighbors to consider when
     *                       determining the score.
     * @param dist           The distance function to use.
     * @param positiveClass  The positive class to use.
     * @return The percentage of newObservation's k newarest neighbors are labeled
     *         with the positiveClass.
     
    public double confidenceScore(Observation newObservation, ArrayList<Observation> trainingSet, int k,
            Distance dist, String positiveClass) {
                

        return 0;
    }
*/

    public static void main(String[] args) throws IOException {
        kNNPA4 knn = new kNNPA4();
        String trainingFilename, testingFilename, distanceFunction, positiveClass;
        int k;
        Dataset trainData, testData;
        Distance distance;

        if(args.length < 5){
            System.err.println(
                "Usage: kNNPA4 <training file> <test file> <k> <distance "+
                    "function> <positive class>\n\n"+
                "<distance function> must be one of:\n"+
                "  * euclidean\n"+
                "  * manhattan\n"+
                "  * cosine\n\n"+
                "<positive class> must : \n"+
                "   * be a string\n"+
                "   * match exaclty 1 of the classes in the dataset.");
            System.exit(1);
        }

        trainingFilename = args[0];
        testingFilename = args[1];
        k = Integer.parseInt(args[2]);
        distanceFunction = args[3];
        positiveClass = args[4];

        distance = knn.new EuclideanDistance();
        if(distanceFunction.equals("manhattan")){
            distance = knn.new ManhattanDistance();
        } else if (distanceFunction.equals("cosine")) {
            distance = knn.new CosineSimilarity();
        }

        // - parse input files
        trainData = knn.parseDataFile(trainingFilename, true);
        testData = knn.parseDataFile(testingFilename, true);


        
        // - run kNN
        // - report the classifications
        //  * test file + one new column: predicted_label
        //System.out.println(testData.columnNamesAsCSV() + ",predicted_label");
        System.out.println(testData.columnNamesAsCSV() + ",predicted_positive");


                

        for(Observation obs : testData.observations)
            System.out.println(obs +","+ String.format("%.5f", 
                    knn.classify(obs, trainData.observations, k, distance, positiveClass)));





    }
}



/**
 * Generate confidence scores which estimate the likelihood that a hypothesis(prediction) is
 * correct.
 * One approach:
– Find features correlated with correctness
– Construct feature vector from good features
– Build correct/incorrect classifier for feature vector.

Given a confidence feature vector we want to classify the vector
as correct or incorrect.

 * 
 *  The objective of the k-NN measures is: To assign higher confidence to
 * those examples that are ‘close’ (i.e. with high similarity) to examples of
 * its predicted class, and are ‘far’ (i.e. low similarity) from examples of a
 * different class. The
 */