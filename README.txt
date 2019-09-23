Running the Algorithms

All of the algorithms, related charts, and cross validation result data can be found at: https://github.com/jagrusy/SupervisedLearning

All of the project requirements are in the `requirements.txt` file

`pip install -r requirements.txt`

The code for each algorithm is contianed in the aptly named files

Running an algorithm (DecisionTree.py, Boosting.py, kNN.py, ANN.py, SVM.py) file will generate the associated
charts which will be saved in the `Figs` folder. In addition, running an algorithm file will generate 2 
pickle files which contain the cross validation results from running Grid Search or Randomized Search. The `.pickle` files are 
stored in the `Pickle` folder.

Each algorithm file utilizes the `util.py` file to gather and preprocess the data. In order for the data to be
processed correctly it should be stored in the `Data` folder.

After running all the algorithms, you can run the `GraphResults.py` file which takes in all the pickle files 
and prints the training scores, testing scores, and accuracy of different algorithms. If you want to look at the pickle
files for a particular algorithm you'll need to specify which algorithms to look at in the `GraphResults.py`
file on line 5. A summary of the results is stored in the `train_test_score_comparison.xlsx` file.

