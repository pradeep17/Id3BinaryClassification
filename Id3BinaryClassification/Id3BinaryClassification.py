#-------------------------------------------------------------------------------
# MACHINE LEARNING - IMPLEMENTATION OF ID3 ALGORITHM FOR BINARY CLASSIFICATION
#-------------------------------------------------------------------------------
# START DATE: 1/15/2014
# END DATE: 1/29/2014
#-------------------------------------------------------------------------------
# NAME : Pradeep Anatharaman(pxa130130)
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
#      Functions        |              Description                                                                                                                        |
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
#| read_input          |    Reads the given file.                                                                      |
#| input               |    items: which is training data in the form of list of lists.                                                                                 |
#| output              |    attributes: which is a list of names of the features.                                                                                       |
#|                  |                                                                                                                                                |
#| find_entropy     |    Calculates the entropy for all Instances that is passes to it based on the class values and returns it.                                     |
#|                     |     classlblcnt stores the counts of each label                                                                                                 |
#|                    |                                                                                                                                                |
#| levelsplitter      |    Splits the iteminstances based on the attribte index and it's value both of which are passed to it and returns the sub iteminstances.       |
#|                     |    newFeatVec stores a list of features without the attribute value that is matched.                                                           |
#|                     |    newX stores the reduced iteminstances.                                                                                                      |
#|                    |                                                                                                                                               |
#| getBestFeature   |    Returns the feature that provides the maximum information gain.                                                                             |
#|                     |    featList stores the feature list for each attribute.                                                                                        |
#|                     |    subiteminstances stores the reduced iteminstances that is returned by levelsplitter function                                                |
#|                    |                                                                                                                                               |
#| retBiggerClass     |    Returns the class label that has majority of the examples.                                                                                  |
#|                     |    classCount dictionary is used to store the count of examples of each class value with class value as key and count as value.                |
#|                     |    checkCompList is passed to check if we should check count of complete list in case of tie among subset of class values for examples at leaf.|
#|                    |                                                                                                                                               |
#| grow_tree         |    It is used to create the decision tree. It return the decision tree that is created in the form of a dictionary.                            |
#|                     |    decisionTree is a dictionary that is used to build the decision Tree recursively by storing another dictionary for next level.              |
#|                     |    attrList stores a list of attribute names.                                                                                                    |
#|                     |    level variable helps in formating the decision tree based on the level of the recursive call                                                |
#|                  |                                                                                                                                               |
#| classifier          |    It is used to classifier a given feature vector and returns the class label that the feature vector belongs to.                               |
#|                     |    cls_list is a list that stores class values of entire iteminstances                                                                        |
#|                    |                                                                                                                                               |
#| calcAccuracy      |    Calculates decision tree accuracy using actual and predicted class labels that are passed in the form of lists                              |
#|                     |    and returns count of correctly classified instances.                                                                                        |
#|                    |                                                                                                                                               |
#|                  |                                                                                                                                                |
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------|


import re
import sys
import operator
from sys import stdout
from math import log

def read_input(input):
    
    
    itemlist = []
    attrlist = []
    content = input.read()
    lines = content.split('\n')
    
    lines.remove(str(lines[0]))         #remove first line of attribute names
    lines = [x for x in lines if x]     #filter bad characters

    itemlen = len(lines[0].split('\t')) - 1  
    for attrvar in range(itemlen):
        attrlist.append("attr"+`attrvar+1`)
    
    for line in lines:
     itemlist.append(line.split('\t'))
    
    return itemlist,attrlist

def find_entropy(iteminstances):
  
 classlblcnt = [0,0]          #1st index represents count of class label '0' and 2nd corresponds to count of class label '1'
 #start counting and calculate entropy for each of the class labels.
 
 for item in iteminstances:
  if item[-1] == '0':
    classlblcnt[0] += 1
  else:
    classlblcnt[1] += 1
   
 result = 0.0
 
 for items in range(1):
  if len(iteminstances) != 0 and classlblcnt[items] !=0:   
   result -= (float(classlblcnt[items])/len(iteminstances)) * log((float(classlblcnt[items])/len(iteminstances)),2)
  else:
   result += 0 
   
 return result

  # Splitting at the current level by using corresponding attribute.
  # Function splits by taking values till the Attribute index and from extending from next value to the current attribute

def levelsplitter(instances,i, attrnum):
 splitinstances = []
 for item in instances:
  if item[attrnum] == i:
    temp = item[:attrnum]
    temp.extend(item[attrnum+1:])
    splitinstances.append(temp)
 return splitinstances


def retBiggerClass(cls_list):
 
 count0=0
 count1=0
            # count0 stores number of instances with class label '0' and count1 corresponds to count of class label '1'
 for item in cls_list:
   if int(item) == 0:
     count0+=1
   else:
     count1+=1
     
 if count1 == count0:                           #In case they are same return -1 to get bigger of entire class labels
     maxcount = -1
 elif max(count0,count1) == count0:             #get max count
     maxcount = '0'
 else:
     maxcount =  '1'
  
 return maxcount
 
def grow_tree(iteminstances,orig_iteminstances,attrList,classitems,level):
 
 
 maxgain = 0.0
 attr_num_selected = -1
 instances = []
 level=level+1
 cls_list = []
 attrColumn = []
 for item in iteminstances:
  cls_list.append(item[-1])        #get all class labels for this instance, negative 1 index returns last item of the list, which is the class label.
  
  #if the Class list contains the same class label for all the instances, return that class label
 if len(cls_list) > 0:
  if cls_list.count(cls_list[0]) == len(cls_list): 
   return cls_list[0]
 
 if len(iteminstances)>0:
  if len(iteminstances[0]) == 1:  #In case no instances are available, get maximum of class count
   maxclass = retBiggerClass(cls_list)
   if maxclass !=-1:
    return maxclass
   else:
    maxclass = retBiggerClass(classitems)            #In case the counts of class labels for this instance are same, get bigger of entire class labels
    if maxclass !=-1:            
     return maxclass
    else:
     return '1'                    #In case they are still same, randomly returning higher class label.
 else:            #do same as above
    maxclass = retBiggerClass(classitems)        
    if maxclass !=-1:
     return maxclass
    else:
     return '1'
     
     
 numOfattr = len(iteminstances[0]) - 1        #total number of attributes in the current instances

 for attrnum in range(numOfattr):
  for item in iteminstances:
   instances.append(item[attrnum])
  attrRange = set(instances)        #getting distinct values from instance set for the range of attribute values to operate on.
  
  #find gain of class variables for this instance set
  gain = find_entropy(iteminstances)

    #iterate over range of distinct attributes and find gain at each.
  for i in attrRange:
   subiteminstances = levelsplitter(iteminstances, i,attrnum )
   gain -= (len(subiteminstances)/float(len(iteminstances))) * find_entropy(subiteminstances)
  
  if (gain > maxgain):
    maxgain = gain
    attr_num_selected = attrnum
   
 attrValue = attrList[attr_num_selected]
 
 
 # using Hash map dictionary data structure to store each level as nodes with attribute value as key and its 
 # instances as values
 
 dictNode = {attrValue:{}}     # hash created with attribute value as key and will store each of the levels as values
 del(attrList[attr_num_selected])
 
 for item in orig_iteminstances:
  attrColumn.append(item[attr_num_selected])
                        
 attrRange = set(attrColumn)            #getting distinct values from instance set for the range of attribute values to operate on from attribute column.
 
 for lbl in attrRange:
  if level==1:
    stdout.write('\n')
  for i in range(0,level-1):
    if i==0 :
     stdout.write('\n')
    stdout.write('| ')
  stdout.write(attrValue + ' = ' + str(lbl) + ' : ')
  newattr = attrList[:]
  dictNode[attrValue][lbl] = grow_tree(levelsplitter(iteminstances, lbl,attr_num_selected),orig_iteminstances,newattr,classitems,level)
 
  if type(dictNode[attrValue][lbl]).__name__ == 'str':                # if we reached leaf with class variable (string type), convert to integer and print
     stdout.write("%d" % int(dictNode[attrValue][lbl]))
     
 return dictNode

        # classification on each item
def classifier(DTree,attributes,iteminstance,classitems):

 keylength=1
 checkClass = False
 root = DTree.keys()[0]     #get top item in the tree passed.
 
 # attribute index of the top node.
 itemnum = attributes.index(root)
 
 key= DTree[root].keys()
 
 # For keys at the next level if a match is found process further
 for key in DTree[root].keys():
  if iteminstance[itemnum] == key:
    checkClass = True
    #if not leaf node continue traversal else get the label.
    if type(DTree[root][key]).__name__!='str':
     classLabel = classifier(DTree[root][key],attributes,iteminstance,classitems)
    else:
     classLabel = DTree[root][key]
  
  elif keylength == len(key) and checkClass == False:
   classLabel = retBiggerClass(classitems)
   if classLabel == -1:
    classLabel = '1'       # if returned class label is -1 i.e no bigger class is found , returning 1 randomly for this binary classification.
  
  keylength+=1

 return classLabel

def findaccuracy(predicted,actual):
    accCount=0
    for i in range(len(actual)):
        if predicted[i] == actual[i]:
         accCount=accCount+1
    return accCount


def main():


    
    trainfile = open(r'train-curve150.dat')
    testfile = open(r'test-3.dat')

    original_train = []
    original_test = []
    predict_train = []
    predict_test = []

    classitems = []



    ##train = open(sys.argv[1])
    ##test = open(sys.argv[2])
    
    # Formating the train and text documents in the form of lists for processing in the algorithm
    train_items,train_attrnames = read_input(trainfile)
    test_items,test_attrnames = read_input(testfile)
    
    trainattrforparse = train_attrnames[:]
    # Sending second copy of training data as feature values need to be iterated on whole training iteminstances and not the reduced iteminstances which would miss out some values in the decision tree.
    id3tree = grow_tree(train_items,train_items,trainattrforparse,classitems,0)
    
    for item in train_items:
     original_train.append(item[-1])
    trainattr = train_attrnames[:]
    classitems = [item[-1] for item in train_items]

    # Generating list of predicted class values using classifier function.
    for item in train_items:
     iteminstances = item[:-1]
     predict_train.append(classifier(id3tree,trainattr,iteminstances,classitems))

    accCount = findaccuracy(predict_train,original_train)

    print '\nAccuracy on training set '+ '(' + str(len(original_train)) + ' instances): ' + str(float(accCount)*100/float(len(original_train))) +' %'
    # Actual class label
    for item in test_items:
     original_test.append(item[-1])
    testattr = test_attrnames[:]
    classitems = [item[-1] for item in train_items]
    # Generating list of predicted class values using classifier function.
    for item in test_items:
     iteminstances = item[:-1]
     predict_test.append(classifier(id3tree,testattr,iteminstances,classitems))
    accCount = findaccuracy(predict_test,original_test)

    print 'Accuracy on test set '+ '(' + str(len(original_test)) + ' instances): ' + str(float(accCount)*100/float(len(original_test))) +' %'



if __name__ == '__main__':
    main()
