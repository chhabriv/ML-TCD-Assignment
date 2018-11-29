# ML-TCD-Assignment

Problem Statement:

102. Dataset Pruning: What is the effect on Machine Learning Performance?

If you prune a dataset, i.e. if you remove certain data that you consider as suboptimal (e.g. all users with less than x ratings), how does that affect the evaluation of machine learning algorithms? For instance, how much “better” is an algorithm becoming when the data is pruned? Or, would an unpruned dataset show that algorithm B is better than algorithm A, but using a pruned dataset would show that algorithm A is better than algorithm B? If you read a research article and the authors report that x% of the original data was removed, how meaningful is the evaluation?

# Feedback from Assignment 1 that can be incorporated:

* Generic improvements what we didn't do as a part of Assignment1 :
1) Calculate statistical significance - confidence interval between results
2) Think of appropriate baseline
3) Outlier detection and dealing with missing values
4) Use cross validation and hold out an extra set for simulating future data
5) Make  more  use  of  the  machine  learning  libraries  features.  It  is  really  easy  to  visualize  datasets,  identify  missing  values,  plot  correlation  heat  maps,  etcwith  scikit-learn  etc.
6) State which library was used for algorithms

* Points to imrpove specific to our team's feedback:
1) Provide more details on features - a table with feature name and type
2) Swap/provide symmetric axis for visualisation of pruned vs target variable
3) Regression is better used for continuous variables, since the measure of incorrectness can be easily weighted, whereas in a     classification, being 1 or 3 measure away for the target is considered the same.
If a regression problem is being converted to classification, the reason should be clearly justified.
4) Use k folder cross validation and provide graph of choosing parameter with respect to k
5) Metrics should also go in methodology. Don't need to explain the formula of standard metrics such as accuracy. If we are then it needs to be in terms of data used for the project. ( Especially  since  you  are  having  a  multi-class  classification  problem,  it  would  be  good  to  explain  how  you  used  accuracy  here  and  also  discuss  it.  For  instance,  you  have  5  classes,  right?  So,  if  the  actual  class  is  3,  and  you  predict  a)  1  and  b  4),  then  both  cases  (a  and  b)  would  be  considered  as  misclassified,  right?  That  is  maybe  not  ideal,  because  in  one  case  you  are  1  rating  off,  and  in  the  other  case  you  are  2  off. )
6) The goal is to discuss results with respect to effect of pruning. No need to discuss which algorithm performs best in general on the dataset.
