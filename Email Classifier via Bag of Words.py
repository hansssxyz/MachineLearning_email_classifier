#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
import scipy.stats
import re #regular expression, used to filter out non-relevant data
import nltk #pacakage to run stemming
import os #operating system
import statistics
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
w_stem=PorterStemmer()

#Here's the path to the dataset
pwd=os.getcwd()
directory1=os.path.join(pwd,"../../enron1/ham")
directory2=os.path.join(pwd,"../../enron1/spam")
files1=os.listdir(directory1)
files2=os.listdir(directory2)
files1f=[os.path.join(directory1, a) for a in files1]
files2f=[os.path.join(directory2, b) for b in files2]
filesall=files1f+files2f
print(len(filesall))
#therefore, all the data in filesall indexed from 0 to 3672 is ham
#therefore, all the rest from 3673 to 5172 is spam
#first I try having 500 testing data and the rest will be training data
#to make a random independent draw, i use random number generator

size=500
testing_index=random.sample(list(range(5172)),size)

data=[]
for index in range(len(filesall)):
    if index not in testing_index:
        with open (filesall[index], 'r',encoding='latin1') as f:
            placeholder=""
            for line in f:
                line=re.sub(r"[^a-zA-Z]"," ",line)
                line=re.sub(r"\s+"," ",line)
                line=line.lower()
                for word in line.split():
                    if len(word)>3:
                        word=w_stem.stem(word)
                        placeholder+=" "+word
        data.append(placeholder)


# In[2]:


bag={}
for s in data:
    for word in s.split():
        if word not in bag.keys():
            bag[word]=1
        else:
            bag[word]=bag[word]+1
words_not_wanted=set()
#For every word that only appears once or twice, we choose to ignore it
for i in bag.keys():
    if bag[i]<=20:
        words_not_wanted.add(i)
for j in words_not_wanted:
    bag.pop(j)
print(len(bag))
l=list(bag.keys())
#then I create a matrix which represents each string in vector form
# dimension: number of columns =len of bag+1, with the last column representing its label
# I use +2 to indicate ham and -2 to indicate spam
# number of rows= number of testing data


matrix=np.zeros((len(filesall)-size,len(bag)+1))
for index in range(len(data)):
    if (index<=3672):
        matrix[index][len(matrix[0])-1]=+2
    else:
        matrix[index][len(matrix[0])-1]=-2                    
    for word in data[index].split():
        if word in l:
            matrix[index][l.index(word)]+=1
print(matrix)           
    


# In[3]:


# Now I do the naive Bayes Case
# first I calculate P(spam) and P(Ham)
count_of_ham=0
for element in matrix:
    if (element[len(element)-1]==2):
        count_of_ham+=1
count_of_spam=len(matrix)-count_of_ham
Prob_ham=count_of_ham/len(matrix)
Prob_spam=1-Prob_ham

#Then I model the conditional probability using 
#1. normal distribution MLE
#2. Exponential Distribution MLE

norm_par_ham=np.zeros((2,len(bag)))
norm_par_spam=np.zeros((2,len(bag)))
exp_par_ham=np.zeros((1,len(bag)))
exp_par_spam=np.zeros((1,len(bag)))

for index in range(len(norm_par_ham[0])):
    sum_ham=0
    sum_spam=0
    for anotherindex in range(len(matrix)):
        if (anotherindex<=count_of_ham-1):
            sum_ham+=matrix[anotherindex][index]
        else:
            sum_spam+=matrix[anotherindex][index]
    norm_par_ham[0][index]=sum_ham/count_of_ham
    norm_par_spam[0][index]=sum_spam/(len(matrix)-count_of_ham)
    if (norm_par_ham[0][index]!=0 and norm_par_spam[0][index]!=0):
        exp_par_ham[0][index]=1/norm_par_ham[0][index]
        exp_par_spam[0][index]=1/norm_par_spam[0][index]
    else:
        #to avoid the case where it's 1/0, we write a large number
        #as a placeholder and skip these cases in our actual classifier
        exp_par_ham[0][index]=99999
        exp_par_spam[0][index]=99999
    
    
for index in range(len(norm_par_ham[0])):
    dev_ham=0
    dev_spam=0
    for anotherindex in range(len(matrix)):
        if (anotherindex<=count_of_ham-1):
            dev_ham+=(matrix[anotherindex][index]-norm_par_ham[0][index])**(2)
        else:
            dev_spam+=(matrix[anotherindex][index]-norm_par_spam[0][index])**(2)
    norm_par_ham[1][index]=np.sqrt(dev_ham/count_of_ham)
    norm_par_spam[1][index]=np.sqrt(dev_spam/(len(matrix)-count_of_ham))
    


# In[4]:


#then, here's the actual classifier. 
def naiveBayes_normal(v):
    #here's the a prior probability
    p_ham=Prob_ham
    p_spam=Prob_spam
    for i in range(len(v)):
        # when the variance is very small PDF can greatly exceed 1,
        # which can skew the data
        if (norm_par_ham[1][i]>0.2 and norm_par_spam[1][i]>0.2):
            p_ham=p_ham*scipy.stats.norm(norm_par_ham[0][i],norm_par_ham[1][i]).pdf(v[i])
            p_spam=p_spam*scipy.stats.norm(norm_par_spam[0][i],norm_par_spam[1][i]).pdf(v[i])
    if(p_ham>p_spam):
        return 2
    else:
        return -2
    
def naiveBayes_exponential(v):
    p_ham=Prob_ham
    p_spam=Prob_spam
    for i in range(len(v)):
        if (exp_par_ham[0][i]<100):
            p_ham=p_ham*exp_par_ham[0][i]*np.exp(-exp_par_ham[0][i]*v[i])
            p_spam=p_spam*exp_par_spam[0][i]*np.exp(-exp_par_spam[0][i]*v[i])
    if(p_ham>p_spam):
        return 2
    else:
        return -2
v=np.zeros(len(matrix[0])-1)
print(naiveBayes_normal(v))
print(naiveBayes_exponential(v))


# In[5]:


#Then I work on the K-Nearest Neighbor.first suppose k=1

#I also attempted to use the divisor by median method but that doesn't seem to work
#because all the medians are essentially 0

#first I do a pre-sorting by filtering out a portion of the dataset
#I randomly pick n features where the given v doesn't return 0, and group all 
#of those training data that doesn't have that feature=0, either

#this method gives a set of n non-empty features
pre_sorting=4
def find_non_empty_features(v):
    nonempty_feature=[]
    for i in range(len(v)):
        if (v[i]!=0):
            nonempty_feature.append(i)
    if (len(nonempty_feature)<pre_sorting):
        return(random.sample(range(len(bag)),pre_sorting))
    feature_index=random.sample(nonempty_feature,pre_sorting)
    return (feature_index)

#this method returns the set of cells where all of the feature indices
#passed in through the argument is non-zero
def find_candidates(list_of_indexfeature):
    candidates=[]
    for i in list_of_indexfeature:
        potential=set()
        for index in range(len(matrix)):
            if (matrix[index][i]!=0):
                potential.add(index)
        candidates.append(potential)
    finalists=candidates[0]
    for i in range(len(candidates)):
        finalists=finalists&candidates[i]
    if (len(finalists)<=5):
        randomadd=random.sample( range(len(bag)),5)
        for i in randomadd:
            finalists.add(i)
    return finalists

def find_1_NN(v):
    s=find_non_empty_features(v)
    finalists=find_candidates(s)
    distance_squared={}
    for index in finalists:
        dist_sq=0
        for j in range(len(bag)):
            dist_sq+=(matrix[index][j]-v[j])**(2)
        distance_squared[index]=dist_sq
    final_index=min(distance_squared.items(),key=lambda x: x[1])[0]
    return matrix[final_index][len(bag)]

#Now I work on the 3_nearest neighbor, the idea is issentially the same
def find_3_NN(v):
    s=find_non_empty_features(v)
    finalists=find_candidates(s)
    distance_squared={}
    for index in finalists:
        dist_sq=0
        for j in range(len(bag)):
            dist_sq+=(matrix[index][j]-v[j])**(2)
        distance_squared[index]=dist_sq
    sorted_distance=sorted(distance_squared.items(),key=lambda x:x[1])
    finalset=[]
    for i in range(min(5,len(sorted_distance))):
        finalset.append(matrix[sorted_distance[i][0]][len(bag)])
    return statistics.mode(finalset)

#Then 5 nearest neighbor is exactly the same except for the last step
def find_5_NN(v):
    s=find_non_empty_features(v)
    finalists=find_candidates(s)
    distance_squared={}
    for index in finalists:
        dist_sq=0
        for j in range(len(bag)):
            dist_sq+=(matrix[index][j]-v[j])**(2)
        distance_squared[index]=dist_sq
    sorted_distance=sorted(distance_squared.items(),key=lambda x:x[1])
    finalset=[]
    for i in range(min(5,len(sorted_distance))):
        finalset.append(matrix[sorted_distance[i][0]][len(bag)])
    return statistics.mode(finalset)


# In[6]:


#here I generate the vector form of the testing data sets
#and then I compare how each classifier performs on the cell below

testing_data=[]
for index in testing_index:
    with open(filesall[index],'r',encoding='latin1') as f:
        placeholder=""
        for lines in f:
            lines=re.sub(r"[^a-zA-Z]"," ",lines)
            lines=re.sub("\s+"," ",lines)
            lines=lines.lower()
            for word in lines.split():
                if (len(word)>=3):
                    word=w_stem.stem(word)
                    placeholder+=" "+word
        if (index<=3672):
            testing_data.append([placeholder,2])
        else:
            testing_data.append([placeholder,-2])
testing_data


testing_matrix=np.zeros((size,len(bag)+1))   
for current in range(size):
    for word in testing_data[current][0].split():
        if word in l:
            testing_matrix[current][l.index(word)]+=1
    if (testing_data[current][1]==2):
        testing_matrix[current][len(bag)]=+2
    else:
        testing_matrix[current][len(bag)]=-2
print(testing_matrix)


# In[7]:


#here I find the accuracy of each 
correct_naiveB_Normal=0
correct_naiveB_Exponential=0
correct_1_NN=0
correct_3_NN=0
correct_5_NN=0

#since finding 5 closest neighbor already finds the 1st and 3 closest neighbor
#to avoid repeating work, I modified my 5_nn to return three values,
#corresponding to the 1NN,3NN,and 5NN

def find_5_NN_modified(v):
    s=find_non_empty_features(v)
    finalists=find_candidates(s)
    distance_squared={}
    for index in finalists:
        dist_sq=0
        for j in range(len(bag)):
            dist_sq+=(matrix[index][j]-v[j])**(2)
        distance_squared[index]=dist_sq
    sorted_distance=sorted(distance_squared.items(),key=lambda x:x[1])
    finalset_1=[]
    finalset_3=[]
    finalset_5=[]
    finalset_1.append(matrix[sorted_distance[0][0]][len(bag)])
    for i in range(min(5,len(sorted_distance))):
        finalset_3.append(matrix[sorted_distance[0][0]][len(bag)])
    for i in range(min(5,len(sorted_distance))):
        finalset_5.append(matrix[sorted_distance[i][0]][len(bag)])
    result=[statistics.mode(finalset_1),statistics.mode(finalset_3),statistics.mode(finalset_5)]
    return result

for i in range(len(testing_matrix)):
    v=np.zeros(len(testing_matrix[0])-1)
    for j in range(len(v)):
        v[j]=testing_matrix[i][j]
    correct=testing_matrix[i][len(bag)]
    if (naiveBayes_normal(v)==correct):
        correct_naiveB_Normal+=1
    if (naiveBayes_exponential(v)==correct):
        correct_naiveB_Exponential+=1
    result_NN=find_5_NN_modified(v)
    if (result_NN[0]==correct):
        correct_1_NN+=1
    if (result_NN[1]==correct):
        correct_3_NN+=1   
    if (result_NN[2]==correct):
        correct_5_NN+=1
accuracy_naiveB_Normal=correct_naiveB_Normal/size
accuracy_naiveB_Exponential=correct_naiveB_Exponential/size
accuracy_1_NN=correct_1_NN/size
accuracy_3_NN=correct_3_NN/size
accuracy_5_NN=correct_5_NN/size
        


# In[8]:


print(accuracy_naiveB_Normal)
print(accuracy_naiveB_Exponential)
print(accuracy_1_NN)
print(accuracy_3_NN)
print(accuracy_5_NN)

