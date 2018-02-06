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

if __name__ == "__main__":
    args = getArgs()
    train_df = pd.read_csv(args.training)
    val_df = pd.read_csv(args.validation)
    test_df = pd.read_csv(args.test)
    attr_list = getAtt(train_df)
    
    #print(train_df.to_string)
    res_list = []
    for attr in attr_list:
        c1 =  train_df[attr].sum()
        c2 = train_df[attr].count() - train_df[attr].sum()
        res = calE(c1,c2)
        res_list.append(res)

    #c1 =  train_df['Class'].sum()
    #c2 = train_df['Class'].count() - train_df['Class'].sum()
    #res = calE(c1,c2)

    testTree = Node(attr_list[0]) #XB
    curNode = testTree 
    curNode.setLeft(Node(attr_list[1])) #XC
    curNode.setRight(Node(attr_list[2])) #XD
    curNode = curNode.getLeft()
    curNode.setLeft(Node("0"))
    curNode.setRight(Node(attr_list[3])) #XE
    curNode = curNode.getRight()
    curNode.setLeft(Node("1"))
    curNode.setRight(Node("0"))
    curNode = testTree.getRight()
    curNode.setLeft(Node("0"))
    curNode.setRight(Node("1"))

    printTree(testTree)
    #print(attr_list)
    #print(res_list)
    #print('Total Class 1: %i' % c1)
    #print('Total Class 0: %i' % c2)
    #print('Entropy of Class Column: %.3f' % res)
