import math
import pandas as pd
from collections import Counter

data = pd.read_csv("../Project#1Data/train.csv")
# print(data)
def entropy(data, target_attribute):
    total=len(data) # total instances 
    count=Counter(data[target_attribute])
    print(count)
    entropy=0
    for val in count.values():
        prob=val/total
        entropy-=(prob)*math.log2(prob)
    return entropy

def weightedGiniIndex(data, feature):
    # giniIndex= 1-Sub(P2);
    total=len(data)

    weightedGini=0
    attributes=data[feature].unique()

    for att in attributes:
        subData=data[data[feature]==att]
        subLen=len(subData)
        values=Counter(subData["isFraud"])
        gini=1
        for val in values.values():
            gini-=math.pow(val/subLen, 2)
        weightedGini+=(subLen/total)*gini

    print(weightedGini)
    return weightedGini

def informationGain(data, feature, attribute):

    #get unique values of the feature
    uniqueVal=data[feature].unique()
    totalLen=len(data)
    print(uniqueVal)
    IG=entropy(data, attribute)
    for val in uniqueVal:
        subData=data[data[feature]==val]
        subLen=len(subData)
        print(subData)
        entropyVal=entropy(subData, attribute)
        IG-=(subLen/totalLen)*entropy
    print(uniqueVal)
    return IG

def findProbability(data, attribute):
    length=len(data)
    count =Counter(data[attribute])
    for k, v in count.items():
        print(k)
        print(v)
        count[k]=(v)/length
    print(count)
    return count        

# def ChiSquare(data, feature, attribute): #attribute is final result

    #step 1. find the probablity of parent node for all outcomes   {val: prob}
    #setp 2. Find the expected yes and no i.e xply count of subset with probability of yes and no
    #step 3. apply formula x2=summ((observed - expected)2/expexted)  // oobserved values are the count of yes and no of subsets

    # total=len(data)
    # count =Counter(data[attribute])

    # values=
    # ExpectedYes=
    
    



# impurity=entropy(data, "isFraud")
findProbability(data, "isFraud")
# informationGain(data, "ProductCD", "isFraud")
# weightedGiniIndex(data, "ProductCD")


