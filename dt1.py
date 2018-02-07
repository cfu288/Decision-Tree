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
def calE(c1,c2):
    c3 = c1+c2
    res = -(c1/c3*np.log2(c1/c3)) - (c2/c3*np.log2(c2/c3))
    return res

def calGain(Es, Ez, Eo , bo, bt):
    return Es - (bo/(bo+bt)*Ez) - (bt/(bo+bt)*Eo)

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

def growTree(examples, target_attribute, attributes):
    numOfOnes = examples["Class"].sum()
    numOfZeros = examples["Class"].count() - numOfOnes
    if numOfZeros == 0: return Node("1")
    elif numOfOnes == 0: return Node("0")
    elif len(attributes) == 0: return  Node("1") if numberOfZeros < numberOfOnes else Node("0")
    else:
        HS = calE(numOfOnes,numOfZeros)
        # assume left branch is when att = 0
        taLeftOnes = examples[("Class" == 1) & (target_attribute == 0)].count() 
        taLeftZeros = examples[("Class" == 0) & (target_attribute == 0)].count()
        taRightOnes = examples[("Class" == 1) & (target_attribute == 1)].count() 
        taRightZeros = examples[("Class" == 0) & (target_attribute == 1)].count()
    return 

if __name__ == "__main__":
    args = getArgs()
    train_df = pd.read_csv(args.training)
    val_df = pd.read_csv(args.validation)
    test_df = pd.read_csv(args.test)
    attr_list = getAtt(train_df)
    
    #print(train_df.to_string)
    res_dict = {}
    # for attr in attr_list:
    #     c1 =  train_df[attr].sum()
    #     c2 = train_df[attr].count() - train_df[attr].sum()
    #     res = calE(c1,c2)
    #     res_dict[attr] = res

    #print(res_dict)
    # GET ROWS WHERE CLASS IS 1
    #print(train_df.loc[train_df["Class"] == 1])
    # GET COUNT OF ROWS WHERE CLASS = 1
    print("SIZE = {}".format(train_df["Class"].loc[train_df["Class"] == 1].count()))
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
