# kNN
k Nearest Neighbors Algorithms - Machine Learning &amp; Artificial Intelligence

PA4: kNN & AUC
===============	


The folder [knn-pa4/] contains the following files and subfolders: 
   1. [datasets/]
       * [iris/]
       * [sms/]
           - [sms_predictions/]  -- To store separately the predictions created by the model for easier use when evaluating.
   2. [java/]
       * [bin/]
       * [src/] 
   3. [README.md]
   4. [evalGraphsExample/] -- contains graphical representations of evaluating k=10, posClass= spam, distance= cosine. 


❏	it describes how to compile (if Java) and run your program

## COMPILE:
First navigate to the [java/] folder: 
  >> ` cd java `

After, compile it:
  >> ` javac -d bin src/*.java ` 

## RUN:

   1. [kNN.java]
    ❏ To run do:
       >> ` java -cp bin kNNPA4 ../datasets/sms/train.csv ../datasets/sms/dev.csv 10 cosine spam `
    
    ❏ To redirect the output of the kNN program to a brand new file, use the redirect operator ` > `. For example: 
       >> ` java -cp bin kNNPA4 ../datasets/sms/train.csv ../datasets/sms/dev.csv 10 cosine spam > ../datasets/sms/sms_predictions/dev_knn_k10_spam_cosine.csv `
    ❏ To see a list of all the options, type: 
       >> ` java -cp bin kNNPA4 `
    ❏ The available command-line arguments for this file are: 
                    <training file> <test file> <k> <distance function> <positive class>

       * <training file>      -- The dataset to be used when training the model (labeled data). 
       * <test file>          -- The dataset used to evaluate the model. Simulates new unseen data. 
       * <k>                  -- The number of nearest neighbors to use when determining the confidence score 
                                    or an observation belonging to the positive class. 
       * <distance function>  -- Computes the distance from a given observation X to its <k> neighbors. 
                                    To then assign a higher confidence score to those observations that are ‘close’ (i.e. with high similarity) to observations of its predicted class, and are ‘far’ (i.e. low similarity) from observations of a different class (negative class).
                                    It supports: euclidean, manhattan & cosine
       * <positive class>  -- The class label to be favored. 


   2. [Evaluation.java]
    ❏ To run do: 
       >> ` java -cp bin Evaluation spam ../datasets/sms/sms_predictions/dev_knn_k10_spam_cosine.csv `
    ❏ To see a list of all the options, type:
       >> ` java -cp bin Evaluation `
    ❏ The available command-line arguments for this file are: 
                    <positive class> <file>
                    
       * <positive class>  -- The class label to be favored.
       * <file>            -- The file that was outputed from the kNN program. 


## EVALUATION SECTION:
Here are the AUC evaluation results of running the kNN program over the SMS 
training and development datasets, with: 
   * k = 10
   * positive class = spam
   with
   * The 3 distance functions available. 

   1. Euclidean result: 0.9896190290554009
   2. Manhattan result: 0.9911720877006558
   3. Cosine result:    0.9896567551763382
As it can be seen, all 3 distance functions produce excelent results. 
But it is the Manhattan one which gives us the best one. 



