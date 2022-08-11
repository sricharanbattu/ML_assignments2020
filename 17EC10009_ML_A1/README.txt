All the programming files are written in python 3.7 on the spyder editor.
Author: Battu SriCharan, Roll Number: 17EC10009, 
email id: sricharanbattu@gmail.com, 
phone: 9553801069, 8942053092
___________________________________________________________________________________________________________________________________________________________________  
1.Where and how are the programs tested?
	The programs are tested on windows 10 command terminal with python 3.7 installed. The programs are also tested on Anaconda PowerShell. (Follow this link for 		installation of python:
	https://realpython.com/installing-python/  if it is already not present). Ideally, the program files should run on any operating system’s command terminal, 		with python 3.7 installed. 
	The command line should be “python filename.py” with additional arguments as stated later.
_____________________________________________________________________________________________________________________________________________________________________

2.Which files are python scripts and which files are module files?
	1.main_file.py, part2.py, part3.py are the script files. 
	2.poly_regression.py, part2_functions.py, part3_functions.py are python modules.

	Thus only main_file.py,part2.py,part3.py should be run to check part 1,2,3 of the assignment respectively. However, all the script and module files should be 		in the same folder.
	There are additional jpeg files, which are named as they are related to that part of the assignment. For instance, all jpeg files consisting of q2a are related 	to question 2, part a in the assignment. 

_____________________________________________________________________________________________________________________________________________________________________
3.Are there any standard python modules imported for the assignment?
	Numpy is imported for computational ease and Matplotlib is imported for plotting. sys module is used for taking command-line arguments. No other standard 	machine learning modules are imported, apart from the modules created by me. Thus the interpreter in which the program is run(ideally command terminals) 		should already have Numpy and Matplotlib installed.  
	_______________________________________________________________________________________________________________________________________________________________________
4.Are there any command-line arguments?
	The command-line arguments are optional. They provide the hyperparameters for  our model. The first command-line argument(threshold) is an integer value 		indicating when to stop the gradient descent. The second command-line argument(alpha) is a float value. It is the learning rate of the gradient descent 		algorithm. A third command-line argument(degree) can be given for the program to be more general and flexible about the maximum degree of the polynomial 		regression model. However, each of these is optional and the default values are 8,0.05,9(as stated in the problem). 

	For instance, when we write the command line “python main_file.py 6 0.04 10”, main_file.py is run. The learning rate of the gradient descent algorithm is 0.04, 	we can build up to 10th- degree polynomial regression model, the gradient descent algorithm stops when the squared euclidian distance between successive theta 		vectors become less than 10^-6. 
_______________________________________________________________________________________________________________________________________________________________________
5.What are the best values for the command line arguments, if any are provided?
	The default values are chosen based on repeated testing of the program, for various command line arguments, so that we don’t need to give up on run time or 		accuracy. A low threshold might achieve very good run time but at a great cost to accuracy. A high threshold might achieve very good accuracy but the run 		time is compromised. In fact, a dramatic shift in accuracy and runtime could be seen when the threshold is changed from 7 to 8. So 7 or 8 is a nearly ideal 		value for threshold, 0.05 for alpha(problem statement) and 9 for degree(problem statement).
_______________________________________________________________________________________________________________________________________________________________________
6.Are there any issues observed while testing?
	1. High threshold(first command-line argument) might achieve a poor run time. 
	2. Also, part2.py and part3.py might take a longer run time as they are built upon main_file.py.  
	3. Each graph that pops up while running should be closed or the program is halted until the plots are closed. 
	4. The errors and their ordering are highly sensitive to the threshold. For instance, the minimum error is obtained for 7th- degree polynomial regression model 	when the threshold is 7, but for the 4th- degree polynomial regression model when the threshold is 8.
_______________________________________________________________________________________________________________________________________________________________________
7.What does each file represent? What could be expected from each file?
	1.  main_file.py  is the solution script for the first part of the assignment. It outputs the plot showing how the train and test data are distributed. It also 	outputs the train and test errors as a function of the degree of the regression model(q1a_train.jpeg,q1a_test.jpeg).
 
	2. poly_regression.py contains the functions employed by main_file.py ( and part3.py). It is a python module and should not be run on its own.

	3. part2.py  is the solution script for the second part of the assignment. It outputs the polynomial fit plots (9 in total q2a_fitdegree_n.jpeg, for each 		degree) and a plot showing how the errors evolve as a function of the degree of the model(q2b_errors.jpeg).

	4. part2_functions.py is a module containing the functions employed by part2.py

	5. part3.py is the solution script for the third part of the assignment. It outputs how the errors for the minimum error model and maximum error model change 		with the regularization parameter in Ridge regression and Lasso. It also outputs the plots showing Lasso and Ridge errors as a function of the regularization 		parameter(q3_MaxErrorModel.jpeg, q3minErrorModel.jpeg)
	6. part3_functions.py is a python module containing the functions employed by part3.py.
_______________________________________________________________________________________________________________________________________________________________________