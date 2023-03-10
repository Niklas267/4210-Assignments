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
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']
test_set = 'contact_lens_test.csv'

def transform_data(db: list):
    
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

for ds in dataSets:

    dbTraining = []
    accuracies = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0: #skipping the header
                dbTraining.append(row)

    #transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    # so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    X, Y = transform_data(dbTraining)

    #loop your training and test tasks 10 times here
    for i in range (10):

        #fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
        clf = clf.fit(X, Y)

        #read the test data and add this data to dbTest
        #--> add your Python code here
        dbTest = []
        with open(test_set, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0: #skipping the header
                    dbTest.append(row)
        
    
        #transform the features of the test instances to numbers following the same strategy done during training,
        #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
        #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
        #--> add your Python code here
        test_X, test_Y = transform_data(dbTest)

        #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
        #--> add your Python code here

        # compare by preicting the whole test sample and counting all hits
        hits = sum(pred == test for (pred, test) in zip(clf.predict(test_X).tolist(), test_Y))
        accuracies.append(hits/len(test_Y))

    #find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here
    avg_acc = sum(accuracies)/len(accuracies)

    #print the average accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print('final accuracy when training on ' +  str(ds) + ': ' + str(avg_acc))




