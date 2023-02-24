#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)


# 0 for -, 1 for +
Y = [float(0) if row.pop() == '-' else float(1) for row in db]
X = [[float(x), float(y)] for x, y in db] 
errors = 0

#loop your data to allow each instance to be your test set
for sample_index in range(len(X)):
    
    testSample = X[sample_index]

    temp_X = X[:sample_index] + X[sample_index + 1:]
    temp_Y = Y[:sample_index] + Y[sample_index + 1:]

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(temp_X, temp_Y)

    #use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    #--> add your Python code here
    prediction = clf.predict([testSample])[0]

    #compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if prediction != Y[sample_index]:
       errors += 1

#print the error rate
#--> add your Python code here
print('error rate:', errors/len(Y))






