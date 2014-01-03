This is a very basic classifier for sentiment analysis of movie reviews.
Classifier Implementation:
-	This classifier implements the ‘Naïve Bayes classifier’ to categorize the movie reviews.  It uses the basic version of MNB model to calculate the probabilistic values.  The ‘train_multinomial_NB’ module takes the features and calculates the model parameters for the MNB classifier. Laplace smoothing has been done in the calculations to eliminate zeros.
-	The program uses the Mutual Information method to perform feature selection.