NAME : BATTU SRI CHARAN
ROLL NO: 17EC10009
email id : sricharanbattu@gmail.com
phone no :9553801069,8942053092

1. Which language is used to write the program?
	python 3.7
2. what are the files present in src folder?
	1.Task1.py is the solution to both questions in Task 1 of the assignment. The result files are stored
	in datasetA(for logistic regression) and datasetB(for decision trees)
	2. Task2_functions.py is essentially a python module,containing the necessary functions for evaluation 
	of the model,for Task2.
	3. Task2_q1_myLogistic.py contains the class definition,functions and everything associated with construction of 
	the model.It answers q1 in Task2.It is a python module.
	4.Task2_q3_mymodelCV.py is a python script, built on Task2_functions.py and Task_2_q1_myLogistic.py.It runs cross validation 
	over datasetA and prints the evaluation metrics. It is the solution to that part of the question related to the model I built,in Task2 q3.
	5. Task2_q23_sklearnCV.py is a python script, built on Task2_functions.py and sklearn. It runs the cross validation over dataseta with
	sklearn's built-in LogisticRegression model.It is the solution to q2 and scikit-learn related question in q3 of Task2.
	6. Task3_functions.py is a python module,containing necessary function definitions for evaluation of
	the model ,for Task3.
	7. Task3_q1_myDecisionTree.py is a python module containing the decision tree(ID3) model built by me.It has necessary class definitions 
	and functions for q1 in Task3.
	8. Task3_q3_mytreeCV.py is a python script ,built on Task3_functions.py and Task3_q1_myDecisionTree.py. It runs crossvalidation over datasetB
	and prints the evaluation metrics. It is the solution to q3 of Task3,relating to the model built without sklearn.
	9. Task3_q23_sklearnCV.py is a python script, built on Task3_functions.py and sklearn.It runs the cross validation over dataseta with
	sklearn's built-in DecisionTree model.It is the solution to q2 and scikit-learn related question in q3 of Task2.

3. How to run the files?
	The files should be run on a command terminal : windows,linux or anaconda powershell. Python3.7 should have been installed. The programs are tested
	in Anaconda PowerShell Prompt on windows10.
	To see the cross validation metrics, one should run :1)Task2_q3_mymodelCV.py for the Logistic Regression model 	without sklearn,
	2)Task2_q23_sklearnCV.py for the Logistic Regression model in sklearn, 3)Task3_q3_mytreeCV.py for the DecisionTree model without sklearn,
	4)Task3_q23_sklearnCV.py for the DecisionTree in sklearn

4. What are the standard libraries required?
	numpy,pandas,sklearn,warnings. These should have been installed for proper running of the files.

5. What are the evaluation metrics observed after cross validation?
	__________________________________________________________________________________
	Model			Accuracy		Precision		Recall
	__________________________________________________________________________________
	1.LogisticRegression	0.8493			0.4644			0.5591
	built from scratch
	(cutoff_prob=0.3)
	
	2.LogisticRegression	0.8680			0.5160			0.5089 
	from sklearn
	(cutoff_prob=0.3)
	_____________________________________________________________________________________
	3.DecisionTree built	0.7892			0.4561			0.4206			
	from scratch					(macroPrecision)	(macroRecall)
	
	4.DecisionTree from	0.8018			0.5396			0.4420
	sklearn						(macroPrecision)	(macroRecall)
	______________________________________________________________________________________

	It is to be noted that Accuracy,Precision and Recall for Logistic Regression are cutoff Probability dependent.
	Metrics for different cutoffs can be seen on running the file Task2_q3_mymodelCV.py and the cutoff could be 
	chosen on user needs. Also, accuracy is not a reliable metric in this case and a compromise between precision
	and recall must be considered. Hence, I chose 0.3 as the cutoff.
	Furthermore, it is not correct to say Logistic Regression is superior to DecisionTree for this data(based on 	
	precision,recall),as they both are doing different tasks. One is a binary classification and the other is a Multiclass Classification.
	