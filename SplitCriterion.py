import math
import pandas as pd
from collections import Counter
import numpy as np
from scipy import stats

data = pd.read_csv("../Project#1Data/train.csv") # Load data from file and create a dataframe
testdata = pd.read_csv("../Project#1Data/test.csv") # Load data from file and create a dataframe
# sampleData =
# print(data)
# def weightBalance(data, target):
#     total =len(data)
#     count = Counter(data[target])
#     classWeights = {
#     0: total / (2 * count[0]),   # non-fraud
#     1: total / (2 * count[1])    # fraud
#     }
#     return classWeights
def entropy(data, target_attribute, IR, majorClass): # define entropy function using features and target attribute
    count=Counter(data[target_attribute]) # make count dict for determining number of instances of target attributes
    total=count[majorClass]*IR+count[not majorClass] # total instances 
    print(count)
    entropy=0 # initialize entropy calculation
    for key, val in count.items(): 
        wt = IR if key==majorClass else 1
        prob=val*wt/(total) # calculate probability for each attributes
        # prob=val/total # calculate probability for each attributes
        entropy-=(prob)*math.log2(prob) # calculate entropy 
    return entropy

def giniIndex(data, attribute, IR, majorClass): #define gini index function using data and attribute as inputs
    # giniIndex= 1-Sub(P2)^2;
    count=Counter(data[attribute]) # Count values agains "isFraud" class
    total=count[majorClass]*IR+count[not majorClass] # total instances 
    gini=1 # initialize weightedGiniIndex calculation
    for key, val in count.items(): # Loop for calulating gini
        wt = IR if key==majorClass else 1
        gini-=math.pow(val*wt/total, 2) # math for gini calculation
    print(gini)
    return gini

def misClassificationError(data, attribute, IR, majorClass):
    # error = 1-max(Pj (for all j))
    count =Counter(data[attribute])
    total=count[majorClass]*IR+count[not majorClass] # total instances 
    maxProb=0
    for key, val in count.items():
        wt = IR if key==majorClass else 1
        prob = val*wt/total
        maxProb=max(maxProb, prob) 
    error =1-maxProb
    return error
        
    

def informationGain(data, feature, attribute, impurityMethod): #Define infromationgain function using data, feature and attribute as inputs

    #get unique values of the feature
    totalLen=len(data)
    #Calculating Imbalance Ratio = countMinorityClass/countMajorityClass
    count=Counter(data[attribute])
    majorityClassCount=max(count[0], count[1]) 
    majorClass=0 if count[0]>count[1] else 1
    minorityClassCount=min(count[0], count[1])
    IR=minorityClassCount/majorityClassCount
    # calculate impurity for parent Node using entrogy, gini-index or misclassificaion error for IG calculation
    IG=entropy(data, attribute, IR, majorClass) if impurityMethod=="entropy" else (giniIndex(data, attribute, IR, majorClass) if impurityMethod=="giniIndex" else misClassificationError(data, attribute, IR, majorClass)) 
    uniqueVal=data[feature].unique() # get unique attibutes from feature
    print(uniqueVal)
    for val in uniqueVal: ## Loop for IG calculation
        subData=data[data[feature]==val] # filter data for unique attribute
        subLen=len(subData)
        print(subData)
         # calcualte impurity for unique feature and filtered data
        childNodeImpurity=entropy(subData, attribute, IR, majorClass) if impurityMethod=="entropy" else (giniIndex(subData, attribute, IR, majorClass) if impurityMethod=="giniIndex" else misClassificationError(subData, attribute, IR, majorClass)) 
        IG-=(subLen/totalLen)*childNodeImpurity # calculate information gain
    print(uniqueVal)
    return IG

def ChiSquare_test(df,Parentfeature,Childfeature,alpha): # Define function to do Chi-Square test using data, Parentfature and Childfeature, alpha as inputs
    Parentunique = list(df[Parentfeature].unique()) # finding unique attributes in parent feature node
    childunique = list(df[Childfeature].unique()) # finding unique attributes in child feature nodes
    DOF = (len(Parentunique)-1)*(len(childunique)-1) # calculate degrees of freedom
    
    ChiMat = pd.crosstab(df[Parentfeature], df[Childfeature]).to_numpy() # calculate Chi-square count matrix
    rowsum = np.sum(ChiMat,axis=1,keepdims=True) # calcuate row sum matrix
    colsum = np.sum(ChiMat,axis=0,keepdims=True) # create column sum matrix
    T = np.sum(colsum) # calcuate total number of counts
    ChiMatExp=np.matmul(rowsum,colsum)/T # create expectation matrix
    Chi_sq=np.sum(((ChiMat-ChiMatExp)**2)/ChiMatExp)# calcuate Chi-square test statistics
    Critical_value = stats.chi2.ppf(1-alpha,DOF) # Find critical Chi-square value for given degree of freedom and alpha value   
    test = 1 # initialize test
    
    if Chi_sq >= Critical_value: # conduct test by comparing with critical value
        test = 0 # Null hypothesis is rejected, DO NOT make parent node a leaf node
    else: 
        test = 1 # Null hypothesis is accepted, make parent node as leaf node
    
    return test
    




# impurity=entropy(data, "isFraud")
# informationGain(data, "ProductCD", "isFraud")
# giniIndex(data, "ProductCD")

# Tree Implementation
class TreeNode:
    def __init__(self, featureAtNode=None, leafPrediction=None, class_counts=None):
        # self.IG=IG #information Gain:: int
        self.featureAtNode=featureAtNode # attribute on which the node will further split
        self.leafPrediction = leafPrediction  # None for a Node 
        self.class_counts=class_counts   #count of classes
        self.children = {} # child is a dictionary of nodes with feature value as key
    def isLeaf(self):
        return self.featureAtNode==None

    def printTree(self, level=0):
            indent = "  " * level
            if self.isLeaf():
                print(f"{indent}Predict → {self.leafPrediction}")
            else:
                print(f"{indent}[Split on: {self.featureAtNode}]")
                for val, child in self.children.items():
                    print(f"{indent}  If {self.featureAtNode} == {val}:")
                    child.printTree(level+2)

#slitting features
def bestSplit(data, features, targetValue, impurityCriteria):
    bestFeature=None
    bestIG=-1
    for feature in features:
        IG=informationGain(data, feature, targetValue, impurityCriteria) #IG method needs data, features, targetValue and ImpurityMethod (entropy, giniIndex, misclassificationError)
        print(IG)
        if(IG>bestIG):
            bestIG=IG
            bestFeature=feature
    return bestFeature, bestIG

#Tree Construction
def constructTree(data, features, target, impurityCriteria, alpha, depth=0, maxDepth=None):
    data = data.copy()
    data=data.dropna(how='all')
    count =Counter(data[target])
    majority=data[target].mode()[0]
    #if pure (all rows have same class)
    if len(data[target].unique())==1:
        return TreeNode(leafPrediction=data[target].iloc[0], class_counts=count)
    #if no feature left/small data
    if not features or (maxDepth and depth>=maxDepth):
        return  TreeNode(leafPrediction=majority, class_counts=count)
    
    #Find the best split using the information Gain
    bestFeature, IG=bestSplit(data, features, target, impurityCriteria)
    #check chi-square for split and to decide further splitting 
    chiSquare=ChiSquare_test(data, bestFeature, target, alpha)  
    if(chiSquare):
        return TreeNode(leafPrediction=majority, class_counts=count)
    #otherwise expand the tree Node
    node =TreeNode(featureAtNode=bestFeature, leafPrediction=majority, class_counts=count)
    for f in data[bestFeature].unique():
        subset=data[data[bestFeature]==f]
        if subset.empty:
            majorityValOfParentNode=data[bestFeature].mode()[0]
            node.children[f]=TreeNode(leafPrediction=majorityValOfParentNode, class_counts=count)
        else:
            remainingFeatures= [f for f in features if f!=bestFeature]
            node.children[f]=constructTree(subset, remainingFeatures, target, impurityCriteria, alpha, depth+1, maxDepth)
    return node    


featureConsidered= ['ProductCD', 'card4', 'card6']
my_tree = constructTree(data, featureConsidered, target="isFraud", impurityCriteria="giniIndex", alpha=0.05, depth=0, maxDepth=None)  # TODO replace the alpha value with correct one

# Print the tree
# if my_tree:
#     my_tree.printTree()


def predictTree(tree, instance):
    node =tree
    while not node.isLeaf():
        feat=node.featureAtNode
        val=instance[feat]
        if pd.isna(val):
            val = "Missing"
        if val in node.children:
            node =node.children[val]
        else:
            return node.leafPrediction
    return node.leafPrediction





    # print(index)
    # print(instance["ProductCD"])
preds = []
for _, instance in testdata.iterrows():
    preds.append(predictTree(my_tree, instance))

# test_preds = pd.Series(preds, index=testdata.index, name="predicted")
output = pd.DataFrame({
    "TransactionID": testdata["TransactionID"],
    "predicted": preds
})
output.to_csv("sampledata.csv", index=False)
print(preds)




