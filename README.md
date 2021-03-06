# Final Project

**Authors:** 
- Danesh Badlani
- Sam Bluestone
- Leslie Le
- Joe Salerno

------------

The goal of this project was to predict student performance using various machine learning methods. The methods employed in this project are *Decision Tree*, *K-Nearest Neighbors*, *Logistic Regression*, *Naive Bayes*, *Random Forest*, and *Support Vector Machine*

You can find the original assignment details [here](https://github.com/lel23/Student-Performance-Prediction/blob/master/Final%20Project%20-%20Data.docx "here"). 
If the link above did not work, you can find the file in the repository under *Final Project - Data.docx* or follow this link: [https://github.com/lel23/Student-Performance-Prediction/blob/master/Final%20Project%20-%20Data.docx](https://github.com/lel23/Student-Performance-Prediction/blob/master/Final%20Project%20-%20Data.docx)

------------

To learn more about this project, please read [ML Final Paper](https://github.com/lel23/Student-Performance-Prediction/blob/master/ML%20Final%20Paper.pdf "*ML Final Paper*").
You can find this file in the repository under *ML Final Paper.pdf*, or you can follow this link: [https://github.com/lel23/Student-Performance-Prediction/blob/master/ML%20Final%20Paper.pdf](https://github.com/lel23/Student-Performance-Prediction/blob/master/ML%20Final%20Paper.pdf)

You can read another paper that conducted research on the same data: [Predict-School-Performance.html](https://github.com/lel23/Student-Performance-Prediction/blob/master/Predict-School-Performance.pdf "Predict-School-Performance.html").
If that link did not work, you can find the file in the repository under *Predict-School-Performance.html* or follow this link: [https://github.com/lel23/Student-Performance-Prediction/blob/master/Predict-School-Performance.pdf](https://github.com/lel23/Student-Performance-Prediction/blob/master/Predict-School-Performance.pdf)

------------

## Things to be Improved
- Number of trees for RFFS is missing
- Random state is not a hyperparameter, as such, it should not be adjusted to attain higher scores in the metrics.
- Cross validation should contain some stratified sampling to ensure balance within each split
- Try regularizing the data with L1 or L2 regularization and testing the models with corresponding data
- Go more in-depth about what was done with Grid Search
- Explain what information we attained from the C values of each model
- Try different ranges for cross validation

------------
Before running any of the files, be sure to install:

- imblearn
- matplotlib
- mlxtend
- numpy
- pandas
- seaborn
- sklearn
