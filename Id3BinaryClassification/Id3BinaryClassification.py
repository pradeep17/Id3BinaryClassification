#-------------------------------------------------------------------------------
# MACHINE LEARNING - IMPLEMENTATION OF ID3 ALGORITHM FOR BINARY CLASSIFICATION
#-------------------------------------------------------------------------------
# START DATE: 1/15/2014
# END DATE: 2/3/2014
#-------------------------------------------------------------------------------
# NAME : Pradeep Anatharaman(pxa130130)
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
#  Functions-           | 
# intput and output     |                                     Description                                                                                                |
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
#| read_input    :      |    Reads the given file line by line and removes bad characters. Parses read values for training and testing instances                    |                                                  |
#| input         :      |    input(file): file to be read and parsed.                                                                                               |
#| output        :      |    itemlist: contains the list of attribute instances read from the file , attrlist: contains the list of attribute names                 |                                                                 |
#|                      |                                                                                                                                           |     
#| find_entropy  :      |    Determines entropy of the given attribute instance set based on class labels.                                                          |
#| input         :      |    iteminstances: attribute instance set.                                                                                                 |
#| output        :      |    result: the calculated entropy.                                                                                                        |
#|                      |                                                                                                                                           |    
#| levelsplitter :      |    Splits the given instances in accordance to the class labels of the given attribute.                                                   |
#| input         :      |    instances:the set of instances to be split, i:the attribute value based on which instances are to be split,                            |
#|                      |    attrnum: the attribute number gives the attribute in consideration                                                                     |
#| output        :      |    splitinstances: the split instances based on the above input.                                                                          |
#|                      |                                                                                                                                           |   
#| retBiggerClass:      |    Finds the larger of the class labels of the instances                                                                                  |
#| input         :      |    cls_list: list of the class labes for the instances in consideration.                                                                  |
#| output        :      |    maxcount: returns the class label which has the maximum count. if cannot decide, returns '1' randomly.                                 |                                                                                                                           |
#|                      |                                                                                                                                           |
#| grow_tree     :      |    grows decission tree (hash data structure) recursively and attribute at each level is determined by the one that has highest           |
#|                      |     information gain.                                                                                                                     |
#|                      |     Once the leaf node is reached prints its value and continues.                                                                         |
#| input         :      |    iteminstances: set of instances under current consideration, orig_iteminstances: set of entire instances maintained for                |
#|                      |     reference for decission on equal splits and information gain, attrList: list of attribute names                                       |
#|                      |     , classitems: set of corresponding class label instances under                                                                        |
#|                      |     consideration, level: maintaining level number to identify the level of the tree we are in.                                           |
#| output        :      |    hashNodes: the hash data table representing the decission tree.                                                                        |                            
#|                      |                                                                                                                                           |
#|                      |                                                                                                                                           | 
#| classifier    :      |     classifies each instance based on the trained decission tree                                                                          |
#| input         :      |      DTree: the trained decission tree, attributes: the list of attribute names, iteminstance: single test instance under consideration   |
#|                      |      ,classitems: the list of class labels from training data set to decide on the larger class, in case decission could not be made      |
#|                      |       on the attributes when there is only one node left with equal gains.                                                                |
#| output        :      |    classLabel: predicted value of the class label for the given instance.                                                                 |        
#|                      |                                                                                                                                           |  
#| findaccuracy  :      |    Finds the accuracy of the predicted class labels based on the count of class labels in the given test and training files               |             
#| input         :      |    classified: count of class labels that have been predicted/classified, original: count of class labels that have been                  |
#|                      |     given in the files.                                                                                                                   |
#| output        :      |    num_match: number of matches between predicted and given values of class labels for the given instance set.                            |                                                           |
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------|


import re
import sys
import operator
from sys import stdout
from math import log

#Reads the given file line by line and removes bad characters. Parses read values for training and testing instances 
def read_input(input):
    
    
    itemlist = []
    attrlist = []
    content = input.read()
    lines = content.split('\n')
    
    lines.remove(str(lines[0]))         #remove first line of attribute names
    lines = [x for x in lines if x]     #filter bad characters

    itemlen = len(lines[0].split('\t')) - 1  
    for attrvar in range(itemlen):
        attrlist.append("attr" + `attrvar + 1`)
    
    for line in lines:
     itemlist.append(line.split('\t'))
    
    return itemlist,attrlist

	#Determines entropy of the given attribute instance set based on class labels.  
def find_entropy(iteminstances):
  
 classlblcnt = [0,0]          #1st index represents count of class label '0' and 2nd corresponds to
                              #count of class label '1'
 #start counting and calculate entropy for each of the class labels.
 
 for item in iteminstances:
  if item[-1] == '0':
    classlblcnt[0] += 1
  else:
    classlblcnt[1] += 1
   
 result = 0.0
 
 for items in range(2):
  if len(iteminstances) != 0 and classlblcnt[items] != 0:   
   result -= (float(classlblcnt[items]) / len(iteminstances)) * log((float(classlblcnt[items]) / len(iteminstances)),2)
  else:
   result += 0 
   
 return result

  # Splitting at the current level by using corresponding attribute.
  # Function splits by taking values till the Attribute index and from
  # extending from next value to the current attribute
 
def levelsplitter(instances,i, attrnum):
 splitinstances = []
 for item in instances:
  if item[attrnum] == i:
    temp = item[:attrnum]
    temp.extend(item[attrnum + 1:])
    splitinstances.append(temp)
 return splitinstances

# Finds the larger of the class labels of the instances
def retBiggerClass(cls_list):
 
 count0 = 0
 count1 = 0
            # count0 stores number of instances with class label '0' and count1
            # corresponds to count of class label '1'
 for item in cls_list:
   if int(item) == 0:
     count0+=1
   else:
     count1+=1
     
 if count1 == count0:                           #In case they are same return -1 to get bigger of
                                                #entire class labels
     maxcount = -1
 elif max(count0,count1) == count0:             #get max count
     maxcount = '0'
 else:
     maxcount = '1'
  
 return maxcount
 
 
 #grows decission tree (hash data structure) recursively and attribute at each level is determined by the one that has highest information gain. 
def grow_tree(iteminstances,orig_iteminstances,attrList,classitems,level):
 
 
 maxgain = 0.0
 attr_num_selected = -1
 instances = []
 level = level + 1
 cls_list = []
 attrColumn = []
 for item in iteminstances:
  cls_list.append(item[-1])        #get all class labels for this instance, negative 1 index returns last
                                   #item of the list, which is the class label.
  
  #if the Class list contains the same class label for all the instances,
  #return that class label
 if len(cls_list) > 0:
  if cls_list.count(cls_list[0]) == len(cls_list): 
   return cls_list[0]
 
 if len(iteminstances) > 0:
  if len(iteminstances[0]) == 1:  #In case no instances are available, get maximum of class count
   maxclass = retBiggerClass(cls_list)
   if maxclass != -1:
    return maxclass
   else:
    maxclass = retBiggerClass(classitems)            #In case the counts of class labels for this instance are same, get
                                                     #bigger of entire class labels
    if maxclass != -1:            
     return maxclass
    else:
     return '1'                    #In case they are still same, randomly returning higher
                                   #class label.
 else:            #do same as above
    maxclass = retBiggerClass(classitems)        
    if maxclass != -1:
     return maxclass
    else:
     return '1'
     
     
 numOfattr = len(iteminstances[0]) - 1        #total number of attributes in the current instances

 for attrnum in range(numOfattr):
  for item in iteminstances:
   instances.append(item[attrnum])
  attrRange = set(instances)        #getting distinct values from instance set for the range of attribute
                                    #values to operate on.
  
  #find gain of class variables for this instance set
  gain = find_entropy(iteminstances)

    #iterate over range of distinct attributes and find gain at each.
  for i in attrRange:
   subiteminstances = levelsplitter(iteminstances, i,attrnum)
   gain -= (len(subiteminstances) / float(len(iteminstances))) * find_entropy(subiteminstances)
  
  if (gain > maxgain):
    maxgain = gain
    attr_num_selected = attrnum
   
 attrValue = attrList[attr_num_selected]
 
 
 # using Hash map dictionary data structure to store each level as nodes with
 # attribute value as key and its
 # instances as values
 
 hashNodes = {attrValue:{}}     # hash created with attribute value as key and will store each of the
                                # levels as values
 del(attrList[attr_num_selected])
 
 for item in orig_iteminstances:
  attrColumn.append(item[attr_num_selected])
                        
 attrRange = set(attrColumn)            #getting distinct values from instance set for the range of
                                        #attribute values to operate on from attribute column.
 
 for lbl in attrRange:
  if level == 1:
    stdout.write('\n')
  for i in range(0,level - 1):
    if i == 0 :
     stdout.write('\n')
    stdout.write('| ')
  stdout.write(attrValue + ' = ' + str(lbl) + ' : ')
  newattr = attrList[:]
  hashNodes[attrValue][lbl] = grow_tree(levelsplitter(iteminstances, lbl,attr_num_selected),orig_iteminstances,newattr,classitems,level)
 
  if type(hashNodes[attrValue][lbl]).__name__ == 'str':                # if we reached leaf with class variable (string type), convert
                                                                       # to integer and print
     stdout.write("%d" % int(hashNodes[attrValue][lbl]))
     
 return hashNodes

        # classification on each item
def classifier(DTree,attributes,iteminstance,classitems):

 keylength = 1
 checkClass = False
 root = DTree.keys()[0]     #get top item in the tree passed.
 
 # attribute index of the top node.
 itemnum = attributes.index(root)
 
 key = DTree[root].keys()
 
 # For keys at the next level if a match is found process further
 for key in DTree[root].keys():
  if iteminstance[itemnum] == key:
    checkClass = True
    #if not leaf node continue traversal else get the label.
    if type(DTree[root][key]).__name__ != 'str':
     classLabel = classifier(DTree[root][key],attributes,iteminstance,classitems)
    else:
     classLabel = DTree[root][key]
  
  elif keylength == len(key) and checkClass == False:
   classLabel = retBiggerClass(classitems)
   if classLabel == -1:
    classLabel = '1'       # if returned class label is -1 i.e no bigger class is found , returning
                           # 1 randomly for this binary classification.
  
  keylength+=1

 return classLabel

 # Finds the accuracy of the predicted class labels based on the count of class labels in the given test and training files    
def findaccuracy(classified,original):
    num_match = 0

    for counter in range(len(original)):
        if classified[counter] == original[counter]:
         num_match += 1
    return num_match

	# main function gets in train data file and test data file as input arguments. If more arguments are given, it throws an error.
	# parses the two files to read and store. Trains the data set and finds accuracy with the training data set.
	# predicts and classifies test data set based on trained decision tree and finds accurance of test data and prints the values.

def main():

    if len(sys.argv) > 3 or len(sys.argv) <= 2:
        print "Invalid number of arguments given. Exactly two input arguments are allowed. Exitting.."
        sys.exit(1)
    
    #trainfile = open('train.dat')
    #testfile = open('test.dat')
    trainfile = open(sys.argv[1])
    testfile = open(sys.argv[2])

    original_train = []
    original_test = []
    predict_train = []
    predict_test = []

    classitems = []

    # Formating the train and text documents in the form of lists for
    # processing in the algorithm
    train_items,train_attrnames = read_input(trainfile)
    test_items,test_attrnames = read_input(testfile)

    trainattrforparse = train_attrnames[:]
    # Sending second copy of training data as feature values need to be
    # iterated on whole training iteminstances and not the reduced
    # iteminstances which would miss out some values in the decision tree.
    id3tree = grow_tree(train_items,train_items,trainattrforparse,classitems,0)

    for item in train_items:
	    original_train.append(item[-1])

    trainattr = train_attrnames[:]      #simple copy of attribute names to trainattr list

    for item in train_items:
	    classitems = [item[-1]]

    # classify training data and find accuracy
    for item in train_items:
	    iteminstances = item[:-1]
	    predict_train.append(classifier(id3tree,trainattr,iteminstances,classitems))
	 
    print '\nAccuracy on training set ' + '(' + str(len(original_train)) + ' instances): ' + str(float(findaccuracy(predict_train,original_train)) * 100 / float(len(original_train))) + ' %'

	#get class labels from test data file for accuracy calculation.
    for item in test_items:
	    original_test.append(item[-1])

    testattr = test_attrnames[:]        #simple copy of attribute names to testattr list

    for item in train_items:
	    classitems = [item[-1]]

    # classify testing data and find accuracy
    for item in test_items:
	    iteminstances = item[:-1]
	    predict_test.append(classifier(id3tree,testattr,iteminstances,classitems))
	 

    print 'Accuracy on test set ' + '(' + str(len(original_test)) + ' instances): ' + str(float(findaccuracy(predict_test,original_test)) * 100 / float(len(original_test))) + ' %'



if __name__ == '__main__':
    main()
