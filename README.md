#-------------------------------------------------------------------------------
# MACHINE LEARNING - IMPLEMENTATION OF ID3 ALGORITHM FOR BINARY CLASSIFICATION
#-------------------------------------------------------------------------------
# START DATE: 1/15/2014
# END DATE: 2/3/2014
#-------------------------------------------------------------------------------
# AUTHOR NAME : Pradeep Anatharaman
# 
#-------------------------------------------------------------------------------
# OPERATING SYSTEM USED: MICROSOFT WINDOWS X64 8.1 
# LANGUAGE USED : PYTHON , INTERPRETER VERSION : 2.7.6 \\Pack available at:http://www.python.org/download/releases/2.7.6/
# ########SOURCE FILES###########
# 1. Id3BinaryClassification.py             - Source file that contains the implementation of ID3 algorithm.
# 
# ########FOLDERS###########
# 1. Data files         - Contains files with training and testing data instances that were provided. 
# 2. Learning curve     - Contains files that were used to plot the learning curve
#########INSTRUCTIONS TO EXECUTE PROGRAM############################
# 1. The source file, training data file and testing data file should be placed in the same folder.
# 2. Open command/shell prompt and navigate to the folder where the above files are placed using "cd" command
# 3. Give command in the following format: Id3BinaryClassification.py <train_file_name> <test_file_name>
#    eg: id3.py train.dat test.dat
#    (refer "Screenshots.pdf" for more examples)
#    IMPORTANT : Ensure that the order of arguments do not change. i.e name of the file that contains training instances
#                should be given first followed by the name of the file that contains testing instances.
#    NOTE      : If the environment variable is set appropriately for Python interpreter, then precede the above command #                  with the keyword "python". i.e python Id3BinaryClassification.py  <train_file_name> <test_file_name> 
# 4. If 2 input arguments are not given(more or less than 2), the program will throw an error and exit.
#
#########ASSUMPTIONS AND NOTES############################
# 1. In cases where decision cannot be made on the classification(Eg. when each attribute in the training instances
#    have same value), then randomly chosen class label is returned(in this program class label '1' is returned).
# 2. Attribute names have been hard coded to begin with string "attr" followed by its number. i.e attr1,attr2 and so on,  #    irrespective of the attribute names given in the train and test data files. The instances have been taken in as is.
#
######### END OF README ###################################
############################ ##############################
############################ ##############################
