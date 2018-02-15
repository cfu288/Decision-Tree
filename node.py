#binary node - left and right branches. More than binary attr will need an array
class Node(object):
    
    def __init__(self, name="", lb=None, rb=None, parent=None):
        self.leftBranch = lb
        self.rightBranch = rb
        self.parent = parent
        self.nodeName = name
        self.numOfOnes = 0
        self.numOfZeros = 0

    def getNumOfOnes(self):
        return self.numOfOnes

    def getNumOfZeros(self):
        return self.numOfZeros

    def setNumOfOnes(self, x):
        self.numOfOnes = x

    def setNumOfZeros(self, x):
        self.numOfZeros = x

    def getVal(self):
        return self.val

    def getLeft(self):
        return self.leftBranch

    def getRight(self):
        return self.rightBranch
    
    def setLeft(self, node):
        self.leftBranch = node
   
    def setRight(self, node):
        self.rightBranch = node

    def getName(self):
        return self.nodeName

    def setName(self, name):
        self.nodeName = name
    
    def setParent(self, parent):
        self.parent = parent

    def getParent(self):
        return self.parent

