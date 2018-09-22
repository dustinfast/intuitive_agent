# Data Sets

This folder contains the following data sets, used for training and validation purposes in the following format:

* No header
* Label in leftmost column
* Feature vector in rightmost colum

## Letter Recognition

From, https://archive.ics.uci.edu/ml/datasets/Letter+Recognition, this set represents 20,000 handwritten digits (one per row) as a set features and labels. Each letter is represented by it's label in the leftmost column and feature vector in the remaining right-hand columns. It has been divided for this application as follows:

* **letter_train.csv**: 15,000 row training set
* **letter_val.csv**: 5,000 row validation set
* **letters.csv**: One row for each letter of the alphabet. Useful for testing.

Example:

``` bash
A,1,1,3,2,1,8,2,2,2,8,2,8,1,6,2,7
B,4,2,5,4,4,8,7,6,6,7,6,6,2,8,7,10
C,7,10,5,5,2,6,8,6,8,11,7,11,2,8,5,9
D,4,11,6,8,6,10,6,2,6,10,3,7,3,7,3,9
...
```

## Word Recognition

TODO