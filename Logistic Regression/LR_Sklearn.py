# -*- coding：utf-8 -*-
# &Author  AnFany

from Heart_Data import model_data  as H_Data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
sklr = LogisticRegression(penalty='l2', tol=10, solver='lbfgs',max_iter=9000)

#格式化输出混淆矩阵
from prettytable import PrettyTable

def confusion(ccmatrix):
    mix = PrettyTable()
    type = sorted(list(range(len(ccmatrix))), reverse=True)
    mix.field_names = [' '] + ['预测:%d类'%si for si in type]
    # 字典形式存储混淆矩阵数据
    for fu in type:
        frru = ['真实:%d类'%fu] + list(ccmatrix[fu][::-1])
        mix.add_row(frru)
    return mix


regre = sklr.fit(H_Data[0], H_Data[1].T[0])

predata = sklr.predict(H_Data[0])
cm = confusion_matrix(H_Data[1].T[0], predata)

print('系数为：\n', sklr.coef_, '\n', sklr.intercept_)



print('混淆矩阵为：\n', confusion(cm))
