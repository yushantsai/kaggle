import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score

# Read files
train_set = pd.read_csv('input/train.csv')
test_set = pd.read_csv('input/test.csv')
validate_set = pd.read_csv('input/gender_submission.csv')

train_rd = train_set.copy()
test_rd = test_set.copy()

# Correct data types
train_rd[['PassengerId','Survived']] = train_rd[['PassengerId','Survived']].astype(object)

# Convert names into titles
train_rd['Name'] = train_rd['Name'].str.replace('.+,\s*(.+?)\..+','\\1',regex=True)
test_rd['Name'] = test_rd['Name'].str.replace('.+,\s*(.+?)\..+','\\1',regex=True)

# Convert cabins into areas
train_rd['Cabin'] = train_rd['Cabin'].str[0]
test_rd['Cabin'] = test_rd['Cabin'].str[0]

train_rd = train_rd.drop(['PassengerId','Ticket'],axis=1)
test_rd = test_rd.drop(['PassengerId','Ticket'],axis=1)

# Remove Cabin because of high ratio of missing value
train_rd = train_rd.drop(['Cabin'], axis=1)
test_rd = test_rd.drop(['Cabin'], axis=1)

# fill in missing values
med = train_rd['Age'].median()
train_rd['Age'] = train_rd['Age'].fillna(med)
test_rd['Age'] = train_rd['Age'].fillna(med)

freq = dict()
for x in train_rd['Embarked']:
    freq[x] = freq.get(x,0)+1

train_rd['Embarked'] = train_rd['Embarked'].fillna(max(freq,key=freq.get))
test_rd['Embarked'] = train_rd['Embarked'].fillna(max(freq,key=freq.get))

for x in test_rd.columns:
    if test_rd[x].dtypes == "object":
        test_rd[x] = test_rd[x].fillna("NAN")
    else:
        test_rd[x] = test_rd[x].fillna(0)

# Labelized caegorical data
lb = LabelBinarizer(neg_label=0,pos_label=1)

for x in train_rd.columns:
    if train_rd[x].dtypes=='object' and x!='Survived':
        lb.fit(train_rd[x])
        train_labels = pd.DataFrame(lb.transform(train_rd[x]),columns=([x+'_'+ y for y in lb.classes_] if lb.classes_.size > 2 else [x]))
        test_labels = pd.DataFrame(lb.transform(test_rd[x]),columns=([x+'_'+ y for y in lb.classes_] if lb.classes_.size > 2 else [x]))

        train_rd = train_rd.drop([x],axis=1)
        test_rd = test_rd.drop([x],axis=1)

        train_rd = train_rd.join(train_labels)
        test_rd = test_rd.join(test_labels)

train_X = train_rd[train_rd.columns[train_rd.columns!='Survived']]
train_y = ['Y' if x==1 else 'N' for x in train_rd['Survived']]

train_X['Name_Capt'] = train_X['Name_Capt'].astype(int)

test_X = test_rd

test_X['Name_Capt'] = test_rd['Name_Capt'].astype(int)

test_y_act = validate_set['Survived']
test_y_act = ['Y' if x==1 else 'N' for x in test_y_act]

# Select the influential columns
cols = dict()

for x in train_X.columns:
    cols[x[:len(x) if x.find('_')<0 else x.find('_')]] = cols.get(x[:len(x) if x.find('_')<0 else x.find('_')],0)+1

ngbr = KNeighborsClassifier(n_neighbors=5)

items = list()
df = pd.DataFrame()
val = 0.0

num = len(cols.keys())

for x in range(num):
    print(str(x+1)+':')
    item = None
    tdf = pd.DataFrame()
    tval = 0.0

    for y in cols.keys():
        if y not in items:
            tdf = pd.concat([df,train_X.filter(like=y)],axis=1)
            ngbr.fit(tdf, train_y)
            acc = accuracy_score(train_y, ngbr.predict(tdf))
            print(y+': '+str(acc))
            if tval < acc:
                item = y
                tval = acc

    if val < tval:
        items.append(item)
        val = tval
        df = pd.concat([df,train_X.filter(like=item)],axis=1)
        print('=> Selected items: ' + ','.join(items)+'\n')
    else:
        print('Done...')
        break

for x in train_X.columns:
    if x[:len(x) if x.find('_')<0 else x.find('_')] not in items:
        train_X = train_X.drop([x], axis=1)
        test_X = test_X.drop([x], axis=1)

# Prediction model
# kNN
model_knn = KNeighborsClassifier(n_neighbors=10)
model_knn.fit(train_X, train_y)
test_y_pred = model_knn.predict(test_X)
acc_knn = round(accuracy_score(test_y_act, test_y_pred),4)
f1_knn = round(f1_score(test_y_act, test_y_pred,pos_label='Y'),4)
prec_knn = round(precision_score(test_y_act, test_y_pred,pos_label='Y'),4)
recall_knn = round(recall_score(test_y_act, test_y_pred,pos_label='Y'),4)

# SVM
model_svm = SVC(kernel='rbf')
model_svm.fit(train_X, train_y)
test_y_pred = model_svm.predict(test_X)
acc_svm = round(accuracy_score(test_y_act, test_y_pred),4)
f1_svm = round(f1_score(test_y_act, test_y_pred,pos_label='Y'),4)
prec_svm = round(precision_score(test_y_act, test_y_pred,pos_label='Y'),4)
recall_svm = round(recall_score(test_y_act, test_y_pred,pos_label='Y'),4)

# Random Forest
model_rf = RandomForestClassifier(n_estimators=10,criterion='entropy')
model_rf.fit(train_X, train_y)
test_y_pred = model_rf.predict(test_X)
acc_rf = round(accuracy_score(test_y_act, test_y_pred),4)
f1_rf = round(f1_score(test_y_act, test_y_pred,pos_label='Y'),4)
prec_rf = round(precision_score(test_y_act, test_y_pred,pos_label='Y'),4)
recall_rf = round(recall_score(test_y_act, test_y_pred,pos_label='Y'),4)

# Logistic Regression
model_lr = LogisticRegression(solver='liblinear')
model_lr.fit(train_X, train_y)
test_y_pred = model_lr.predict(test_X)
acc_lr = round(accuracy_score(test_y_act, test_y_pred),4)
f1_lr = round(f1_score(test_y_act, test_y_pred,pos_label='Y'),4)
prec_lr = round(precision_score(test_y_act, test_y_pred,pos_label='Y'),4)
recall_lr = round(recall_score(test_y_act, test_y_pred,pos_label='Y'),4)

# submission = pd.DataFrame({
#         'PassengerId':validate_set['PassengerId'],
#         'Survived':[1 if x=='Y' else 0 for x in test_y_pred]
#     })
#
# submission['Survived'] = submission['Survived'].astype(int)
#
# submission.to_csv('output/submission.csv',index=False)

performance = pd.DataFrame({
    'Model':['KNN','Support Vector Machines','Random Forest','Logistic Regression'],
    'ACC':[acc_knn,acc_svm,acc_rf,acc_lr],
    'F1':[f1_knn,f1_svm,f1_rf,f1_lr],
    'Precision':[prec_knn,prec_svm,prec_rf,prec_lr],
    'Recall':[recall_knn,recall_svm,recall_rf,recall_lr]})

print(performance)
