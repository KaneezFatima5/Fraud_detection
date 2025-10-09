import math
import pandas as pd
from collections import Counter
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

data = pd.read_csv("train.csv") # Load data from file and create a dataframe
testdata = pd.read_csv("test.csv") # Load data from file and create a dataframe


def processNumericData(data, numericFeatures, skew=1.0, topFreqThresh =0.5, nBins=5):
    processedData=data.copy()
    processedData=processedData.replace("NotFound", np.nan) #replacing not found with nan for data imputation
    for feature in numericFeatures:
        processedData[feature] = pd.to_numeric(processedData[feature], errors="coerce") 

        #replace NAN with median values
        processedData[feature]=processedData[feature].fillna(processedData[feature].median())
        try:
            #Use Noramlized frequencies instead of normal freq
            topFreq=processedData[feature].value_counts(normalize=True, dropna=True).iloc[0] #in descending order
        except IndexError:
            continue
        if abs(processedData[feature].skew()) > skew or topFreq>topFreqThresh:
            #Using Quntile binning for highly skewed data, where top freq > 0.5
            try:
                processedData[feature]=pd.qcut(processedData[feature], q=nBins, duplicates='drop')
            except ValueError: #if the unique values are very few then skip binning
                processedData[feature]=processedData[feature]
        else:
            processedData[feature]=processedData[feature].astype(float)
    return processedData

def entropy(data, target_attribute, classWeights): # define entropy function using features and target attribute
    count=Counter(data[target_attribute]) # make count dict for determining number of instances of target attributes
    totalWt = sum(count[c] * classWeights.get(c, 1.0) for c in count)  #getting weighted total 
    entropy=0 # initialize entropy calculation
    for key, val in count.items(): 
        prob = (val * classWeights.get(key, 1.0)) / totalWt # calculate weighted probability for each attributes
        # prob=val/total # calculate probability for each attributes
        if prob >0: entropy-=(prob)*math.log2(prob) # calculate entropy 
    return entropy

def giniIndex(data, target, classWeights): #define gini index function using data and attribute as inputs
    # giniIndex= 1-Sub(P2)^2;
    count=Counter(data[target]) # Count values agains "isFraud" class
    totalWt = sum(count[c] * classWeights.get(c, 1.0) for c in count)
    gini=1 # initialize weightedGiniIndex calculation
    for key, val in count.items(): # Loop for calulating gini
        prob = (val * classWeights.get(key, 1.0)) / totalWt  # calculate weighted probability for each attributes
        gini-=math.pow(prob, 2) # math for gini calculation
    return gini

def misClassificationError(data, target, classWeights):
    count = Counter(data[target])
    if not count:
        return 0.0  # No data → no impurity
    totalWt = sum(count[c] * classWeights.get(c, 1.0) for c in count)
    if totalWt == 0:  #safety check for empty subset
        return 0.0
    max_p = max((count[c] * classWeights.get(c, 1.0)) / totalWt for c in count) #iterate over each class to get max probability 
    return 1 - max_p
        
    

def informationGain(data, feature, target, impurityMethod, classWeights): #Define infromationgain function using data, feature and attribute as inputs

    #get unique values of the feature
    totalLen=len(data)
    if totalLen==0: return 0.0
    #Calculating Imbalance Ratio = countMinorityClass/countMajorityClass
    count=Counter(data[target])
    if 0 not in count or 1 not in count: return 0.0
    # calculate impurity for parent Node using entrogy, gini-index or misclassificaion error for IG calculation
    parentImpurity=entropy(data, target, classWeights) if impurityMethod=="entropy" else (giniIndex(data, target, classWeights) if impurityMethod=="giniIndex" else misClassificationError(data, target, classWeights)) 
    featureVal=data[feature] # get attibutes from feature
    if pd.api.types.is_numeric_dtype(featureVal):        #check if the feature is numeric  
        sorted_values=np.sort(featureVal.unique()) 
        if len(sorted_values)<=1: return 0
        thresholds=(sorted_values[:-1] +sorted_values[1:])/2  #calculate threshold values/average value for each instance
        bestIG=-float("inf")

        for t in thresholds:
            left=data[featureVal<=t] #if data is less or equal to the threshold val
            right=data[featureVal>t]
            if len(left)==0 or len(right)==0:
                continue
            leftImpurity=entropy(left, target, classWeights) if impurityMethod=="entropy" else (giniIndex(left, target, classWeights) if impurityMethod=="giniIndex" else misClassificationError(left, target,classWeights)) 
            rightImpurity=entropy(right, target,  classWeights) if impurityMethod=="entropy" else (giniIndex(right, target, classWeights) if impurityMethod=="giniIndex" else misClassificationError(right, target, classWeights)) 

            weightedChildNodeImpurity=(len(left)/totalLen*leftImpurity + len(right)/totalLen *rightImpurity)
            IG=parentImpurity-weightedChildNodeImpurity
            bestIG=max(bestIG, IG)
        return bestIG
    uniqueVal=featureVal.unique()
    childImpurity=0.0
    for val in uniqueVal: ## Loop for IG calculation
        subData=data[data[feature]==val] # filter data for unique attribute
        subLen=len(subData)
        # calcualte impurity for unique feature and filtered data
        childNodeImpurity=entropy(subData, target, classWeights) if impurityMethod=="entropy" else (giniIndex(subData, target, classWeights) if impurityMethod=="giniIndex" else misClassificationError(subData, target, classWeights)) 
        childImpurity+=(subLen/totalLen)*childNodeImpurity # calculate information gain
    IG=parentImpurity-childImpurity
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
    

#slitting features
def bestSplit(data, features, targetValue, impurityCriteria, classWeights):
    bestFeature=None
    bestIG=-1
    #randomize the features at each split
    randomFeatures=np.random.choice(features, size=int(np.sqrt(len(features))), replace=False)
    #check IG for each feature and get the best feature
    for feature in randomFeatures:
        IG=informationGain(data, feature, targetValue, impurityCriteria, classWeights) #IG method needs data, features, targetValue and ImpurityMethod (entropy, giniIndex, misclassificationError)
        if(IG>bestIG):
            bestIG=IG
            bestFeature=feature
    return bestFeature, bestIG


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

#Tree Construction
def constructTree(data, features, target, impurityCriteria, alpha, depth=0, maxDepth=None, classWeights=None):
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
    bestFeature, IG=bestSplit(data, features, target, impurityCriteria, classWeights)
    #check chi-square for split and to decide further splitting 
    chiSquare=ChiSquare_test(data, bestFeature, target, alpha)  
    if(chiSquare):
        return TreeNode(leafPrediction=majority, class_counts=count)
    #otherwise expand the tree Node
    node =TreeNode(featureAtNode=bestFeature, leafPrediction=majority, class_counts=count)
    for f in data[bestFeature].unique():
        subset=data[data[bestFeature]==f]
        if subset.empty:
            #calculate class with highest freq and save it as leaf node
            majorityValOfParentNode=data[bestFeature].mode()[0]
            node.children[f]=TreeNode(leafPrediction=majorityValOfParentNode, class_counts=count)
        else:
            #filter out already utilized feature
            remainingFeatures= [f for f in features if f!=bestFeature]
            #recursively call construct tree until base conditions are met
            node.children[f]=constructTree(subset, remainingFeatures, target, impurityCriteria, alpha, depth+1, maxDepth, classWeights)
    return node    



class randomForest:
    def __init__(self, data, features=None, numberOfTrees=None, sampleSize=None, target="isFraud", impurityCriteria="giniIndex", alpha=0.05, maxDepth=None):
        self.data=data
        self.features=features
        self.numberOfTrees=numberOfTrees
        self.sampleSize=sampleSize
        self.target=target
        self.impurityCriteria=impurityCriteria
        self.alpha=alpha
        self.depth=0
        self.maxDepth=maxDepth
        self.classWeights=self.computeClassWeights()
        self.trees=[]

    def computeClassWeights(self):
        ##Compute inverse-frequency class weights.
        counts = Counter(self.data[self.target])
        total = sum(counts.values()) or 1
        return {cls: total / (len(counts) * counts[cls]) for cls in counts}
    
    def balancedBootstrap(self):
        #####Return a balanced bootstrap sample with equal classes ####
        df_major = self.data[self.data[self.target] == 0]
        df_minor = self.data[self.data[self.target] == 1]
        n_major = len(df_major)
        n_minor = len(df_minor)

        # target number of samples per class = min class count × sampleSize
        n_per_class = int(min(n_major, n_minor) * self.sampleSize)
        if n_per_class == 0:
            return self.data.sample(frac=self.sampleSize, replace=True)

        sample_major = df_major.sample(n=n_per_class, replace=True)
        sample_minor = df_minor.sample(n=n_per_class, replace=True)
        sample = pd.concat([sample_major, sample_minor]).sample(frac=1.0, replace=False)
        return sample.reset_index(drop=True)
    def randomTree(self):
        for i in range(self.numberOfTrees):
            # sample data with equal number of classes
            sample = self.balancedBootstrap()
            randomFeatures=list(np.random.choice(self.features, size=max(1, int(len(features)/3)), replace=False))
            tree=constructTree(sample, randomFeatures, target=self.target, impurityCriteria=self.impurityCriteria, alpha=self.alpha, depth=self.depth, maxDepth=self.maxDepth, classWeights=self.classWeights)
            self.trees.append(tree)

    def predictTree(self, instance):
        randomTreeResult=[]
        randomTrees=self.trees
        for rTree in randomTrees:
            node =rTree
            while not node.isLeaf():
                feat=node.featureAtNode
                val=instance[feat]
                if pd.isna(val):
                    val = "Missing"
                if val in node.children:
                    node =node.children[val]
                else:
                    break
            randomTreeResult.append(node.leafPrediction)
        majorityVote=Counter(randomTreeResult).most_common(1)  #[(class, occurance)]
        return majorityVote[0][0]
    
    def predictAll(self, testData):
        preds =[]
        for _, instance in testData.iterrows():
            preds.append(self.predictTree(instance))
        # test_preds = pd.Series(preds, index=testdata.index, name="predicted")
        output = pd.DataFrame({
            "TransactionID": testData["TransactionID"],
            "predicted": preds
        })
        output.to_csv("sampledata.csv", index=False)

def Confusion_mat_results(Prediction,Validation,Zscore):  ### Define a fucntion that create confusion matrix and Mean Misclassification error
    df = pd.DataFrame({'Validation':Validation,'Prediction':Prediction}) ## Take two numpy array as input for prediction and validation data and make a dataframe out of them
    Match_data = pd.crosstab(df['Validation'],df['Prediction'],dropna=True) ## Make cross table for combination of True and False for prediction
    Confusion_mat = Match_data.to_numpy() ## Convert the table to numpy array

    TP = Confusion_mat[1,1] ## True positive count
    FP = Confusion_mat[1,0] ## True negative count
    FN = Confusion_mat[0,1] ## False negative count
    TN = Confusion_mat[0,0] ## True Negative count
     
    N = TP+FP+FN+TN ## number of samples
    MME = (FP+FN)/N  ## Mean Misclassification error
    Acc = (TP+TN)/N  ## Accuracy
    FNR = FN/(TP+FN) ## False Negative Rate
    FPR = FP/(TN+FP) ## False Position Rate
    BER = 0.5*(FNR+FPR) ### Balanced Error Rate
    BAcc = 1-BER ### Balanced Accuracy
    SE = np.sqrt(BER*(1-BER)/N) ## Standard Error
    Cinterval = [BER-Zscore*SE, BER+Zscore*SE] ## Confidence Interval
    
    return dict({'MME':MME,'BER':BER,'Cinterval_Low':Cinterval[0],'Cinterval_High':Cinterval[1]}) ## Return the results in a dictionary


################################# HYPER_PARAM CONTROL AND FUNCTION CALLS ###################################################################################
categoricalFeatures= ['ProductCD', 'card4', 'card6']
numericalFeatures=["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "addr2", "card3", "card5"]

data=processNumericData(data, numericFeatures=numericalFeatures, skew=1.0, topFreqThresh=0.5, nBins=8)
testdata=processNumericData(testdata, numericFeatures=numericalFeatures, skew=1.0, topFreqThresh=0.5, nBins=8)
features = ["ProductCD", "card1", "card2", "card3", "card4", "card5", "card6", "addr2", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14"]  #
# impurityCriteria=["misClassificationError", "entropy", "giniIndex"]
forest=randomForest(data, features=features, numberOfTrees=5, sampleSize=0.95, target="isFraud", impurityCriteria="giniIndex", alpha=0.05, maxDepth=13)
forest.randomTree()

for i, tree in enumerate(forest.trees):  # print first 2 trees for brevity
    print(f"\nTree {i+1}:")
    tree.printTree()


forest.predictAll(testData=testdata)


