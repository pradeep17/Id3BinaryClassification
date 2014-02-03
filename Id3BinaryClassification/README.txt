#-------------------------------------------------------------------------------
# MACHINE LEARNING - IMPLEMENTATION OF ID3 ALGORITHM FOR BINARY CLASSIFICATION
#-------------------------------------------------------------------------------
# START DATE: 1/15/2014
# END DATE: 2/3/2014
#-------------------------------------------------------------------------------
# NAME : Pradeep Anatharaman(pxa130130)
# EMAIL: pxa130130@utdallas.edu
#-------------------------------------------------------------------------------
# OPERATING SYSTEM USED: MICROSOFT WINDOWS X64 8.1 
# LANGUAGE USED : PYTHON , INTERPRETER VERSION : 2.7.6 \\Pack available at:http://www.python.org/download/releases/2.7.6/
# ########FILES###########
# 1. id3.py             - Source file that contains the implementation of ID3 algorithm.
# 2. README.TXT         - This file.
# 3. LearningCurve.pdf  - The file contains graph of the learning curve based on accuracy plotted on training and 
#						  test data instances.
# 4. Screenshots.pdf    - The file contains screenshots taken during testing and execution of the program.
# ########FOLDERS###########
# 1. Data files         - Contains files with training and testing data instances that were provided. 
# 2. Learning curve     - Contains files that were used to plot the learning curve

#########INSTRUCTIONS TO EXECUTE PROGRAM############################
# 1. The source file, training data file and testing data file should be placed in the same folder.
# 2. Open command/shell prompt and navigate to the folder where the above files are placed using "cd" command
# 3. Give command in the following format: id3.py <train_file_name> <test_file_name>
#    eg: id3.py train.dat test.dat
#    (refer "Screenshots.pdf" for more examples)
#    IMPORTANT : Ensure that the order of arguments do not change. i.e name of the file that contains training instances
#                should be given first followed by the name of the file that contains testing instances.
#    NOTE      : If the environment variable is set appropriately for Python interpreter, then precede the above command with
#				 the keyword "python". i.e python id3.py <train_file_name> <test_file_name> 
# 4. If more than 2 input arguments are given, the program will throw an error and exit.
#
######### END OF README ###################################
############################ ##############################
############################ ##############################