#!/usr/bin/env python3
#binary node - left and right branches. More than binary attr will need an array
class Node(object):
    
    def __init__(self, name, lb=None, rb=None, parent=None):
        self.leftBranch = lb
        self.rightBranch = rb
        self.parent = parent
        self.nodeName = name or ""

    def getVal(self):
        return self.val

    def getLeft(self):
        #if self.leftBranch != None:
        return self.leftBranch
        #else:
        #    err = "leftBranch is None on node {}".format(self.nodeName)
        #    raise ValueError(err)

    def getRight(self):
        #if self.rightBranch != None:
        return self.rightBranch
        #else:
        #    err = "rightBranch is None on node {}".format(nodeName)
        #    raise ValueError(err)
    
    def setLeft(self, node):
        self.leftBranch = node
   
    def setRight(self, node):
        self.rightBranch = node

    def getName(self):
        return self.nodeName
