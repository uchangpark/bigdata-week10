# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import pandas as pd
import numpy as np

tennis_data = pd.read_csv('playtennis.csv')
tennis_data

tennis_data.Outlook = tennis_data.Outlook.replace('Sunny', 0)
tennis_data.Outlook = tennis_data.Outlook.replace('Overcast', 1)
tennis_data.Outlook = tennis_data.Outlook.replace('Rain', 2)
tennis_data.Temperature = tennis_data.Temperature.replace('Hot', 3)
tennis_data.Temperature = tennis_data.Temperature.replace('Mild', 4)
tennis_data.Temperature = tennis_data.Temperature.replace('Cool', 5)
tennis_data.Humidity = tennis_data.Humidity.replace('High', 6)
tennis_data.Humidity = tennis_data.Humidity.replace('Normal', 7)
tennis_data.Wind = tennis_data.Wind.replace('Weak', 8)
tennis_data.Wind = tennis_data.Wind.replace('Strong', 9)
tennis_data.PlayTennis = tennis_data.PlayTennis.replace('No', 10)
tennis_data.PlayTennis = tennis_data.PlayTennis.replace('Yes', 11)
tennis_data

X = np.array(pd.DataFrame(tennis_data, columns = ['Outlook', 'Temperature', 'Humidity', 'Wind']))
y = np.array(pd.DataFrame(tennis_data, columns = ['PlayTennis']))
X_train, X_test, y_train, y_test = train_test_split(X, y)

gnb_clf = GaussianNB() # Gaussian Naive Bayes 
gnb_clf = gnb_clf.fit(X_train, y_train)
gnb_prediction = gnb_clf.predict(X_test)
# sample date test...
my_test=[[1,5,6,8]] # 흐리고, 춥고, 습도가높고, 바람이 약한날 테이스를 칠까?
my_test_result= gnb_clf.predict(my_test)
print(my_test_result)
"""
# Naive Bayes 모델의 predict함수를 사용해 X_test 데이터에 대한 예측값과 실제값 y_test를 비교해 모델의 성능을 평가하겠습니다.
# 성능 평가에 사용될 평가 요소는 confusion_matrix, classification_report, f1_score, accuracy_score입니다.
# 성능 평가를 하기 위해 sklearn.metrics 패키지의 confusion_matrix, classification_report, f1_score, accuracy_score 모듈을 import합니다.
"""
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
'''
Confusion Matrix는 오차행렬을 나타냅니다. Confusion Matrix의 결과를 보면 2x2 행렬인 것을 알 수 있습니다.
Confusion Matrix의 y축은 실제값, x축은 예측값입니다.
'''

print('Confusion Matrix')
print(confusion_matrix(y_test, gnb_prediction))
'''
Classification Report는 분류에 대한 측정 항목을 보여주는 보고서를 나타냅니다.
Classification Report의 측정 항목으로는 클래스 별의 precision, recall, f1-score와
전체 데이터의 precision, recall, f1-score가 있습니다.
'''

print('Classification Report')
print(classification_report(y_test, gnb_prediction))

# 실제값과 예측값에 f1-score함수를 사용해 구한 f-measure와 accuracy_score 함수를 사용해 구한 accuracy를 나타내보겠습니다.

'''
f1_score 함수에 파라미터로 실제값 y_test와 예측값 gnb_prediction을 넣고 average를 weighted로 설정합니다. 
weighted는 클래스별로 가중치를 적용하는 역할을 합니다. 이렇게 3개의 파라미터를 넣고 f1_score를 구한 후 
round 함수를 이용해 소수점 2번째 자리까지 표현한 값을 변수 fmeasure에 저장합니다.
'''
fmeasure = round(f1_score(y_test, gnb_prediction, average = 'weighted'), 2)
'''
accuracy_score 함수에 파라미터로 실제값 y_test와 예측값 gnb_prediction을 넣고 normalize를 True로 설정합니다.
True는 정확도를 계산해서 출력해주는 역할을 합니다. False로 설정하게 되면 올바르게 분류된 데이터의 수를 출력합니다.
이렇게 3개의 파라미터를 넣고 accuracy를 구한 후 round 함수를 이용해 소수점 2번째 자리까지 표현한 값을 변수 accuracy에 저장합니다.
'''
accuracy = round(accuracy_score(y_test, gnb_prediction, normalize = True), 2)
# 컬럼이 Classifier, F-Measure, Accuracy인 데이터프레임을 변수 df_nbclf에 저장합니다.
df_nbclf = pd.DataFrame(columns=['Classifier', 'F-Measure', 'Accuracy'])
'''
컬럼 Classifier에는 Naive Bayes로 저장하고, 데이터프레임 df_nbclf에 loc 함수를 사용해 
컬럼에 맞게 fmeasure 데이터와 accuracy 데이터를 데이터프레임에 저장합니다.
'''
df_nbclf.loc[len(df_nbclf)] = ['Naive Bayes', fmeasure, accuracy]
# 저장한 데이터프레임을 출력합니다.
df_nbclf
