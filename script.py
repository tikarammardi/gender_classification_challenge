from sklearn import tree, neighbors, discriminant_analysis,naive_bayes

clf = tree.DecisionTreeClassifier()

# CHALLENGE - create 3 more classifiers...
# 1 GaussianNB
clf1 = naive_bayes.GaussianNB()

# 2 KNeighborsClassifier
clf2 = neighbors.KNeighborsClassifier()

# 3 QuadraticDiscriminantAnalysis
clf3 = discriminant_analysis.QuadraticDiscriminantAnalysis()

#[height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# Train them
clf = clf.fit(X,Y)

clf1 = clf1.fit(X,Y)

clf2 = clf2.fit(X,Y)

clf3 = clf3.fit(X,Y)

prediction = clf.predict([[190, 70, 43]])
probability = clf.predict_proba([[190, 70, 43]])

prediction1 = clf1.predict([[190,70,43]])
probability1 = clf1.predict_proba([[190, 70, 43]])

prediction2 = clf2.predict([[190,70,43]])
probability2 = clf2.predict_proba([[190, 70, 43]])

prediction3 = clf3.predict([[190,70,43]])
probability3 = clf3.predict_proba([[190, 70, 43]])

#compare their results and print the best one


#print(prediction)

print(probability1)
print(prediction1)

print(probability2)
print(prediction2)

print(probability3)
print(prediction3)
