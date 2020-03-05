# LT2212 V20 Assignment 2

Put any documentation here including any answers to the questions in the
assignment on Canvas.

Documentation

Part 1

The pipeline for tokenization used in my assignment():

split with space -> lower -> length control -> stopwrods crontrol -> only alphabetic -> lemmatization

Part 2
Truncated SVD is used for the original version of a2.py

Part 3

classifier equals GaussianNB (if id = 1)
           equals DecisionTreeClassifier (if id = 2)

WEIGHTED AVG CLF1 0.31
WEIGHTED AVG CLF2 0.24

Part 4

total features available: 7551
WEIGHTED AVG CLF1 0.68 0.67 0.67
WEIGHTED AVG CLF2 0.61 0.61 0.61

N-Dim = 1000
WEIGHTED AVG CLF1 0.32 0.15 0.14
WEIGHTED AVG CLF2 0.31 0.31 0.31

N-Dim = 100
WEIGHTED AVG CLF1 0.37 0.16 0.15
WEIGHTED AVG CLF2 0.30 0.30 0.30

N-Dim = 50
WEIGHTED AVG CLF1 0.37 0.19 0.18
WEIGHTED AVG CLF2 0.30 0.30 0.30

N-Dim = 25
WEIGHTED AVG CLF1 0.33 0.18 0.16
WEIGHTED AVG CLF2 0.30 0.30 0.31

N-Dim = 10
WEIGHTED AVG CLF1 0.28 0.14 0.13
WEIGHTED AVG CLF2 0.24 0.24 0.24

N-Dim = 5
WEIGHTED AVG CLF1 0.20 0.13 0.09
WEIGHTED AVG CLF2 0.16 0.16 0.16

Due to running time, the reporting dimetions are restricted as above.
The accuracy reduces in general with the reducing number of dimentions. However, the performance reducation is rather trivial, the trade off between computing time and accuracy is to be considered. When the demention is then reduced to less than total categories (20), there are not enough number of dimention to support classification performance, great reduction in performance can be observed.


Bonus Part
New dimention reduction algorithm used: PCA

N-Dim = 1000
WEIGHTED AVG CLF1 0.32 0.16 0.14
WEIGHTED AVG CLF2 0.30 0.30 0.30

N-Dim = 100
WEIGHTED AVG CLF1 0.33 0.16 0.15
WEIGHTED AVG CLF2 0.31 0.30 0.31

N-Dim = 50
WEIGHTED AVG CLF1 0.36 0.20 0.18
WEIGHTED AVG CLF2 0.31 0.31 0.31

N-Dim = 25
WEIGHTED AVG CLF1 0.33 0.14 0.13
WEIGHTED AVG CLF2 0.32 0.32 0.32

N-Dim = 10
WEIGHTED AVG CLF1 0.29 0.14 0.12
WEIGHTED AVG CLF2 0.23 0.23 0.23

P.S. I discussed with Axel in details about the relatively low performance of the
classifier. After comparing the effects of different classifier. I found out that
the reason of the low performance can be traced back to the classifier I used. Leaniar
classifier and decision trees are not best suitable for the task.
