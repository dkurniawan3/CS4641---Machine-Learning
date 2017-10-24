Daniel Kurniawan
903043201

CS 4641 - Machine Learning

Datasets:
Both datasets are attached in the submission.

Wine Quality - White
https://archive.ics.uci.edu/ml/datasets/wine+quality
(only need to download the winequality-white.csv)

Pen Digits
https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits
I downloaded both the testing and training zip files (.Z), unzipped and converted to csv,
and combined them into one csv. 

Running the Code:
To run the code, you need Python 2.7.* and above and need to have the following dependencies installed:
- pandas
- numpy
- matplotlib
- sklearn (scikit-learn)

I created a utility script called helper.py with two helper functions taken from sklearn's documentation
to help plot confusion matrices and learning curves of the various machine learning algorithms I implemented.

Simply call 'python WQSupervisedLearning.py' to run all algorithms on the wine quality dataset and
'PDSupervisedLearning.py' to run all algorithms on the pen digit dataset. The results, graphs, etc should be
automatically generated. One thing to note - plotting in matplotlib can block the flow of code so you'll need
to exit out of graphs in order to continue running the ML applications.
