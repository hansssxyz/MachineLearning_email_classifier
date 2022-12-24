# email_classifier

Overview:

part A) Introduction and Background to project
Part B) Code Explaination
Part C) Analysis of the performance of each classifier 
Part D) References

A)
This project is aiming at the classfication of emails into spam and non-spam(ham) by implementing machine learning models. First, all the emails are embedded in high-dimensional Eucleadian space by using the bag of words model. Simply put, this model splits each email into individual words and count the number of appearances of each word. Then, I wrote four classifiers that I've learned in the machine learning course at Columbia (CS 4771), namely, the naives bayers classifiers trained with different probablistic models, and the k-nearest neighbor model. In the former, I used maximum likelihood estimate assuming gaussian and exponential distribution, and for the latter, I tried 1, 3, and 5 nearest neighbor. Then I ran a training set of ~4000 emails, and tested them on ~500 emails. The data set is provided by professor Verma.

B)

I have a total of 7 cells of code, I will give a brief overview of each.

In cel1 1, I imported operating system and used that package to create a path from my jupyternotebook location to the location where my file is actually downloaded. Then I filtered out the data using re(regular expression) and nltk(naturla language toolkit) to pre-process the data.

In cell 2, I created a bag of words by first creating a dictionary, and then going through every single word of every line to get all the words in the dictionary. After that, I eliminated all the words that only appeared once or twice, and then I converted every single email into a vector in the eucledian space, each of dimention approximately 5000-10000(I first did over 10000,but then my computer is too slow 

In cell 3, I performed the naive Bayes classifier. First I attempted a normal distribution and used the MLE to find the system parameters. However, since a lot of data(each individual words) have a uniformly 0 distribution, the standaard deviation is often very small, which means that the Probability Density Functino could be very large in some places. So I added an extra condition while calculating the class conditional: only work with those stdev greater than 0.2. Then I worked on the naive Bayes classifier but with exponential distribution. Since MLE for exponential is the inverse of system mean, which is 0 in a lot of columns, I also chose to ignore those invalid data.

Cell 4 is the actual implementation of the Naive Bayes Normal/Exponential method

In cell 5, I worked on the k nearest neighbor(K NN). Since the dimensionality is too large for a standard k NN algorithm, I implemented a pre-processing method where given each new test data, I pick 2 features randomly from all the non-zero features. Then I pick my K NN candidates from all the training vectors which also have those two features non-zero. I could have eliminated even more data by picking 3, 4, or 5 non-zero features, but given the limited size of data, i won't want to over process it.

Then in the set of potential vectors, I calculated the square of distance for each, and found the minimum vectors by using sorted(list,key: lambda x:x[1], and finding the label of the first k entires.

The last two cells, cell 6 and cell 7, are just processing the testing data and working to find the accuracy of each classifier. 

C) 

In my training data,I picked 100, 500, and 1000 testing samples to see how each of my 5 classifier(naive bayes normal, naive bayes exponential, 1, 3, and 5 nearest neighbor performs given the data set. In the 1000 testing sample case, I limited my bag of words to -4000 words and my KNN to picking the candidate set with at least 4 non-empty similar label between the candidates and the testing point. 

Overall, the 1,3,5 nearest neighbor all performed reasonably well, with around 80 accuracy. If more time could be allowed, I would have tested my candidates in my nearest neighbor analysis, and in fact it is very time-consuming. The naive bayes normal also did fine, and I can still play around with the parameters more to determine what to do with very large standard deviations. Lastly, the naive bayes exponential model just fails. Please see the  graph below to check the performance of each of the three classfier. 

<img width="658" alt="Screen Shot 2022-12-24 at 8 56 29 AM" src="https://user-images.githubusercontent.com/121346627/209439003-a5d3634c-f46e-41ab-ab01-b11758056400.png">

D)
References:
https://www.youtube.com/watch?v=GFGxeqPR_bU&list=PLk0inNMEQQPXQCNAMq59Vwhmnn3sItHFk&index=10
I received help from Andrew Yang and Elise Han with accessing the email txt documents and creating the bag of words. 
