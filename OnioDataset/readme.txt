Data Preprocessing & Feature Extraction - PRIYANKA HARESHBHAI SORATHIYA - k67900
The dataset (OnionOrNot.csv) is loaded, and missing values are removed. Headlines are converted into numerical features using TF-IDF vectorization, which highlights important words while filtering out common ones. The dataset is then split into training (75%) and testing (25%) sets for model evaluation.  

Model Training & Evaluation - BILAL BILAL - k67963
Several machine learning models (*Random Forest, SVC, Naïve Bayes, Logistic Regression) are trained and evaluated using cross-validation to determine the best performer. *Multinomial Naïve Bayes (NB) is selected for its effectiveness in text classification and trained on the complete dataset.  

Model Testing & Predictions - ANUSHA VISHWANATH SALIMATH - k67910
The trained model is tested on unseen data, generating a classification report and a confusion matrix for performance analysis. Finally, a new headline is classified to demonstrate real-world application, predicting whether it belongs to The Onion (satirical) or Not The Onion (real news).
