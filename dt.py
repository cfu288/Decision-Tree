import pandas as pd
import numpy as np
import argparse
from node import Node

def getArgs():
    p = argparse.ArgumentParser(description='Simple decision tree implementation')
    p.add_argument('training', help='training set csv file location')
    p.add_argument('validation', help='validation set csv file location')
    p.add_argument('test', help='test set csv file location')
    p.add_argument('toprint', help='yes/no print tree')
    p.add_argument('prune', help='yes/no prune tree')
    return p.parse_args()

#Calc Entropy
def calE(c1,c2): #c1 is no 1's, c2 is no 0's
    c3 = c1+c2
    res = 0
    if (c1 == 0) and (c2 == 0):
        res = 0
    elif (c1 == 0):
        res = 0 - (c2/c3*np.log2(c2/c3))
    elif (c2 == 0):
        res = -(c1/c3*np.log2(c1/c3)) - 0
    else:
        res = -(c1/c3*np.log2(c1/c3)) - (c2/c3*np.log2(c2/c3))
    return res

def calGain(Es, Ez, Eo , bz, bo):
    return Es - ((bz/(bz+bo))*Ez) - ((bo/(bo+bz))*Eo)

def calVarImp(c1,c2): #c1 is no 1's, c2 is no 0's
    res = 0
    if (c1 == 0) or (c2 == 0):
        res = 0
    else:
        res = c1/(c1+c2) * c2/(c1+c2)        
    return res

def getAtt(df):
    l = list(df.columns.values)
    l.remove('Class') # rm class attr
    return l

def printTree(root, level=0):
    curLevel = "|" * level
    leaf = 0
    if root.getLeft() == None and root.getRight() == None:
        leaf = 1
        print(root.getName(),end="")
    if leaf == 0:
        print("\n{}{} = 0 : ".format(curLevel,root.getName()),end="")
    if root.getLeft() != None:
        printTree(root.getLeft(),level+1)
    
    if leaf == 0: 
        print("\n{}{} = 1 : ".format(curLevel,root.getName()),end="")
    if root.getRight() != None:
        printTree(root.getRight(),level+1)

def growTree(examples, attributes,parent=None):
    root = Node()
    root.setParent(parent)
    numOfOnes = examples["Class"].sum()
    numOfZeros = examples["Class"].count() - numOfOnes
    root.setNumOfOnes(numOfOnes)
    root.setNumOfZeros(numOfZeros)
    if numOfZeros == 0:
        root.setName("1")
        return root
    elif numOfOnes == 0:
        root.setName("0")
        return root
    elif len(attributes) == 0: 
        root.setName("1") if numOfZeros < numOfOnes else root.setName("0")
        return root
    else:
        best_attr = getBestAttr(examples, attributes)
        root.setName(best_attr)
        new_attr_list = attributes[:] # new copy of list for recursion
        new_attr_list.remove(best_attr)

        left_examples_subset = examples.loc[examples[best_attr] == 0]
        left_examples_count = examples["Class"].loc[examples[best_attr] == 0].count()
        if left_examples_count == 0:
            leaf = Node()
            leaf.setParent(root)
            leaf.setName("1") if numOfZeros < numOfOnes else leaf.setName("0")
            root.setLeft(leaf) 
        else:
            # recurse on left
            root.setLeft(growTree(left_examples_subset, new_attr_list,root))

        right_examples_subset = examples.loc[examples[best_attr] == 1]
        right_examples_count = examples["Class"].loc[examples[best_attr] == 1].count()
        if right_examples_count == 0:
            leaf = Node()
            leaf.setParent(root)
            leaf.setName("1") if numOfZeros < numOfOnes else leaf.setName("0")
            root.setRight(leaf) 
        else: 
            # recurse on right
            root.setRight(growTree(right_examples_subset, new_attr_list,root))
    
    return root

def getBestAttr(examples, attributes):
    numOfOnes = examples["Class"].sum()
    numOfZeros = examples["Class"].count() - numOfOnes
    HS = calE(numOfOnes,numOfZeros) # total of current set, unrealated to target attr
    maxDict={}
    maxList=[]
    nameEquivList=[]
    for attr in attributes:
        taLeftZeros = examples["Class"].loc[(examples["Class"] == 0) & (examples[attr] == 0)].count()
        taLeftOnes = examples["Class"].loc[(examples["Class"] == 1) & (examples[attr] == 0)].count() 
        taRightZeros = examples["Class"].loc[(examples["Class"] == 0) & (examples[attr] == 1)].count()
        taRightOnes = examples["Class"].loc[(examples["Class"] == 1) & (examples[attr] == 1)].count() 
        HSvLeft = calE(taLeftOnes,taLeftZeros) 
        HSvRight = calE(taRightOnes, taRightZeros)
        attr_gain = calGain(HS,HSvLeft,HSvRight, taLeftOnes+taLeftZeros, taRightOnes+taRightZeros)
        maxDict[attr] = attr_gain
        maxList.append(attr_gain)
        nameEquivList.append(attr)
    # print(maxDict)  
    return nameEquivList[maxList.index(max(maxList) )]
    #return max(maxDict, key=maxDict.get)

def testTree(treeRoot, testData):
    currentRows = 0
    numberCorrect = 0
    for row in testData.itertuples():
        currentRows += 1
        # recursively check if test row matches tree path
        numberCorrect += testTreeHelper(treeRoot, row)
    return numberCorrect/currentRows

def testTreeHelper(treeRoot, row):
    if treeRoot == None:
        print("ERR, none node when testing")
        return
    if (treeRoot.getName() == "0"):
        if(row.Class == 0): return 1
        else: return 0
    elif (treeRoot.getName() == "1"):
        if(row.Class == 1): return 1
        else: return 0
    else:
        # get the node in tree and check path
        currentNode = treeRoot.getName()
        path = getattr(row,currentNode)
        if path == 0:
            return testTreeHelper(treeRoot.getLeft(), row)
        elif path == 1:
            return testTreeHelper(treeRoot.getRight(), row)
        else:
            print("ERR, path does not exist")

def getLeafSet(treeRoot):
    leafSet = set()
    getLeafSetHelper(treeRoot,leafSet)
    return leafSet

def checkIfLastLay(node):
    if ((node.getLeft().getName() == "0") or (node.getLeft().getName() == "1")) and ((node.getRight().getName() == "0") or (node.getRight().getName() == "1")):
        return 1
    return 0

def getLeafSetHelper(treeRoot,leafSet):
    if (treeRoot.getLeft() == None) and (treeRoot.getRight() == None):
        if (treeRoot.getParent() != None): 
            if (checkIfLastLay(treeRoot.getParent())):
                leafSet.add(treeRoot.getParent())
        return
    else:
        getLeafSetHelper(treeRoot.getLeft(), leafSet)
        getLeafSetHelper(treeRoot.getRight(), leafSet)

def pruneTree(treeRoot, val_df):
    # test Validation set on tree - save initial percentage
    currentHigh = testTree(treeRoot, val_df)
    # get set of nodes above leaf nodes
    set = getLeafSet(treeRoot)
    # set up empty dict -> for node:percentage
    nodePercentages = {}
    # for node in leaf node set
    for node in set:
        #save left branch pointer
        leftPtr = node.getLeft()
        #save right branch pointer
        rightPtr = node.getRight()
        #turn branches into None - node is a leaf
        node.setLeft(None)
        node.setRight(None)
        # need to set name of node to new 0 or 1 based on percent
        oldName = node.getName()
        node.setName("0") if node.getNumOfZeros() > node.getNumOfOnes() else node.setName("1")
        #Test validation set on new tree - add node:percentage to dict
        newHigh = testTree(treeRoot, val_df)
        nodePercentages[node] = newHigh
        #revert left and right branches
        node.setLeft(leftPtr)
        node.setRight(rightPtr)
        node.setName(oldName)
    #get max percentage in dictionary 
    newMaxNode = max(nodePercentages, key=nodePercentages.get)
    # if max percentage > initial percentage
    print("PRUNING? {} new{} old{}".format(newMaxNode.getName(),nodePercentages[newMaxNode] ,currentHigh))
    if nodePercentages[newMaxNode] >= currentHigh:
        # turn node with max percentage into leaf for realz
        print("PRUNING? {}".format(newMaxNode.getName()))
        newMaxNode.setLeft(None)
        newMaxNode.setRight(None)
        newMaxNode.setName("0") if newMaxNode.getNumOfZeros() > newMaxNode.getNumOfOnes() else newMaxNode.setName("1")
        # recurse on new pruned tree
        pruneTree(treeRoot,val_df)
    # else stop, further pruneing wont improve percentage
    return 

if __name__ == "__main__":
    args = getArgs()
    train_df = pd.read_csv(args.training)
    val_df = pd.read_csv(args.validation)
    test_df = pd.read_csv(args.test)
    attr_list = getAtt(train_df)

    treeRoot = growTree(train_df, attr_list)
    
    res = testTree(treeRoot, test_df)
    
    print("Accuracy is {:.3f}%".format((res)*100))
    
    pruneTree(treeRoot,val_df)
    
    res = testTree(treeRoot, test_df)
    
    print("Pruned accuracy is {:.3f}%".format((res)*100))

     

    #printTree(treeRoot)
        #print("itm: {}".format(item.getName()))
    #res = getBestAttr(train_df,attr_list)
    #print("max:{}".format(res)) 
    
    #examples = train_df
    #best_attr = getBestAttr(examples, attr_list)
    #print(best_attr)
    #left_examples_subset = examples.loc[examples[best_attr] == 0]
    
    #newlist = attr_list[:]
    #newlist.remove(best_attr)
    #print(attr_list)
    #print(newlist)
    
    #print(left_examples_subset)
    #best_attr1 = getBestAttr(left_examples_subset, newlist)
    
    # print(left_examples_subset)
    # sub1 = examples.loc[examples[attr_list[0]] == 0]
    # print(sub1)
    # sub2 = sub1.loc[sub1[attr_list[1]] == 0]
    # print(sub2)
    
    # print(train_df.to_string)
    # res_dict = {}
    # for attr in attr_list:
    #     c1 =  train_df[attr].sum()
    #     c2 = train_df[attr].count() - train_df[attr].sum()
    #     res = calE(c1,c2)
    #     res_dict[attr] = res

    #print(res_dict)
    # GET ROWS WHERE CLASS IS 1
    #print(train_df.loc[train_df["Class"] == 1])
    # GET COUNT OF ROWS WHERE CLASS = 1
    # print("SIZE = {}".format(train_df["Class"].loc[train_df["Class"] == 1].count()))
    # testTree = Node(attr_list[0]) #XB
    # curNode = testTree 
    # curNode.setLeft(Node(attr_list[1])) #XC
    # curNode.setRight(Node(attr_list[2])) #XD
    # curNode = curNode.getLeft()
    # curNode.setLeft(Node("0"))
    # curNode.setRight(Node(attr_list[3])) #XE
    # curNode = curNode.getRight()
    # curNode.setLeft(Node("1"))
    # curNode.setRight(Node("0"))
    # curNode = testTree.getRight()
    # curNode.setLeft(Node("0"))
    # curNode.setRight(Node("1"))
    # printTree(testTree)
    #print(attr_list)
    #print(res_list)
    #print('Total Class 1: %i' % c1)
    #print('Total Class 0: %i' % c2)
    #print('Entropy of Class Column: %.3f' % res)
