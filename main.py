from sklearn import tree
from sklearn import svm
from sklearn.linear_model import SGDClassifier
#training data
#dataset - height, weight, shoe size
X = [[181, 80, 11], [177, 70, 10], [160, 60, 7], [154, 54, 6], [166, 65, 7],
     [190, 90, 14], [175, 64, 8],
     [177, 70, 9], [159, 55, 6], [171, 75, 9], [181, 85, 10]]
#target - male or female 
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


#decision tree
clf_tree = tree.DecisionTreeClassifier()
clf_tree = clf_tree.fit(X,Y)
print(clf_tree.predict([[180, 86, 10]]))

#SVM(Support Vector Machines)
clf_svm = svm.SVC()
clf_svm = clf_svm.fit(X,Y)
print(clf_svm.predict([[180, 86, 10]]))

#SGD(Stochastic Gradient Descent)
clf_sgd = SGDClassifier()
clf_sgd = clf_sgd.fit(X,Y)
print(clf_sgd.predict([[190, 86, 10]]))
