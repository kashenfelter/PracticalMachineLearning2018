---
title: 'Practical Machine Learning Peer-Graded Assignment: Prediction Assignment Writeup'
author: "Kathy Targowski Ashenfelter"
date: "June 12, 2018"
output:
  html_document:
    df_print: paged
    keep_md: TRUE
---
###Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

#Preparation
Load the necessary packages and read in the data

```r
#rm(list=ls())               
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## 
## Attaching package: 'caret'
```

```
## The following object is masked from 'package:httr':
## 
##     progress
```

```r
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
```

```
## Rattle: A free graphical interface for data science with R.
## Version 5.1.3 Copyright (c) 2006-2017 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(randomForest)
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:rattle':
## 
##     importance
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
library(knitr)
Link1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
Link2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
training <- read.csv(url(Link1), na.strings=c("NA","#DIV/0!",""))
testing <- read.csv(url(Link2), na.strings=c("NA","#DIV/0!",""))
```
###Exploratory Anlaysis of the Data


```r
dim(training)
```

```
## [1] 19622   160
```

```r
dim(testing)
```

```
## [1]  20 160
```

```r
table(training$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

This dataset contains 19622 observations (rows) and 160 variables (columns) in training dataset. The last column is the category variable classe. The most frequenctly-occuring class is A.

In preparation for the analysis, I remove missing values and extraneous variables.


```r
NA_Count = sapply(1:dim(training)[2],function(x)sum(is.na(training[,x])))
NA_list = which(NA_Count>0)
colnames(training[,c(1:7)])
```

```
## [1] "X"                    "user_name"            "raw_timestamp_part_1"
## [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
## [7] "num_window"
```

```r
training = training[,-NA_list]
training = training[,-c(1:7)]
training$classe = factor(training$classe)
testing = testing[,-NA_list]
testing = testing[,-c(1:7)]
```
###Cross-Validation
Since this exercise is inherently a classification task, I utilized two f the built-in classification methods available in the *caret* package- the classification tree algorithm and random forest. Additionally, I performed a 3-fold cross-validation with caret's trainControl function.


```r
set.seed(1234)
crossfold3 = trainControl(method="cv",number=3,allowParallel=TRUE,verboseIter=TRUE)
modrf = train(classe~., data=training, method="rf",trControl=crossfold3)
```

```
## + Fold1: mtry= 2 
## - Fold1: mtry= 2 
## + Fold1: mtry=27 
## - Fold1: mtry=27 
## + Fold1: mtry=52 
## - Fold1: mtry=52 
## + Fold2: mtry= 2 
## - Fold2: mtry= 2 
## + Fold2: mtry=27 
## - Fold2: mtry=27 
## + Fold2: mtry=52 
## - Fold2: mtry=52 
## + Fold3: mtry= 2 
## - Fold3: mtry= 2 
## + Fold3: mtry=27 
## - Fold3: mtry=27 
## + Fold3: mtry=52 
## - Fold3: mtry=52 
## Aggregating results
## Selecting tuning parameters
## Fitting mtry = 27 on full training set
```
##Cross-Validation

```r
modtree = train(classe~.,data=training,method="rpart",trControl=crossfold3)
```

```
## + Fold1: cp=0.03568 
## - Fold1: cp=0.03568 
## + Fold2: cp=0.03568 
## - Fold2: cp=0.03568 
## + Fold3: cp=0.03568 
## - Fold3: cp=0.03568 
## Aggregating results
## Selecting tuning parameters
## Fitting cp = 0.0357 on full training set
```
Next, we evaluate the performance of these two models using the testing dataset.


```r
prf=predict(modrf,training)
ptree=predict(modtree,training)
table(prf,training$classe)
```

```
##    
## prf    A    B    C    D    E
##   A 5580    0    0    0    0
##   B    0 3797    0    0    0
##   C    0    0 3422    0    0
##   D    0    0    0 3216    0
##   E    0    0    0    0 3607
```


```r
table(ptree,training$classe)
```

```
##      
## ptree    A    B    C    D    E
##     A 5080 1581 1587 1449  524
##     B   81 1286  108  568  486
##     C  405  930 1727 1199  966
##     D    0    0    0    0    0
##     E   14    0    0    0 1631
```
Using the testing data: 


```r
prf=predict(modrf,testing)
ptree=predict(modtree,testing)
table(prf,ptree)
```

```
##    ptree
## prf A B C D E
##   A 7 0 0 0 0
##   B 3 0 5 0 0
##   C 0 0 1 0 0
##   D 0 0 1 0 0
##   E 1 0 2 0 0
```

This model testing indicates that the random forest model is the most accurate for testing dataset.

###Conclusion

Since it was the most accurate for the testing data in my exploratory analysis, I used the random forest model on the testing dataset for my final submission analysis.

```r
answers=predict(modrf,testing)
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
answers
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
pml_write_files(answers)
```

The above output represents the  predicted classes for the 20 tests: 
