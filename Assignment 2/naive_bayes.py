#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv, copy

#reading the training data in a csv file
#--> add your Python code here
db = []
db_test = []

#reading the data in a csv file
with open('weather_training.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header and remove day
         db.append (row[1:])

#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
def transform_list(db: list):
    # Convert each colmun into its numerical rep
    for column in range(len(db[0])):
        # get all data from column
        curr_column = [row[column] for row in db]
        # get distinct values
        curr_values = list(set(curr_column))
        curr_values.sort()
        # replace each value with the numerical representation
        for row, old_value in enumerate(curr_column):
            db[row][column] = curr_values.index(old_value)

    X = db
    Y = [row.pop() for row in X]
    return X, Y

X, Y = transform_list(db)
#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the test data in a csv file
#--> add your Python code here
#reading the data in a csv file
with open('weather_test.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db_test.append(row)

#printing the header os the solution
#--> add your Python code here
format_string = "{:<5}{:<10}{:<13}{:<10}{:<8}{:<12}{:<12}"
print(format_string.format('Day', 'Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis', 'Confidence'))

#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
# Convert each colmun into its numerical rep
test_X, test_Y = transform_list(copy.deepcopy(db_test))

for index, row in enumerate(test_X):
    curr = db_test[index]
    conf = round(clf.predict_proba([row[1:5]]).max(), 2)
    result = 'Yes' if clf.predict([row[1:5]])[0] == 1 else 'No'
    if conf >= 0.75:
        print(format_string.format(curr[0] ,curr[1], curr[2], curr[3], curr[4], result, conf))


