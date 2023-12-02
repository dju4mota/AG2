import pandas as pd

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as dc

df = pd.read_csv("tic_tac_toe.csv")
df.head()

data = pd.DataFrame(df)
data = data.replace("b", "0")
data = data.replace("x", "1")
data = data.replace("o", "-1")
data = data.replace("positivo", "1")
data = data.replace("negativo", "-1")

X = data.iloc[:, 0:9]
Y = data.iloc[:, 9]
print(X)
print(Y)

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2)

model = dc()
model.fit(X, Y)

classifier = DecisionTreeClassifier()
classifier = classifier.fit(xtrain, ytrain)
ypred = classifier.predict(xtest)
print('Accuracy Score:', metrics.accuracy_score(ytest, ypred))

flag = True
while flag:
    print("Entre com os valores do tabuleiro, 'x', 'b' ou 'o', separados por vírgulas: (entre com -1 para sair)")
    entrada = input()
    lista = [str(i) for i in entrada.split(',')]
    if lista[0] == '-1':
        break

    newdata_N = []
    error = False
    for i in range(9):
        if lista[i] == 'o':
            newdata_N.append(-1)
        elif lista[i] == 'x':
            newdata_N.append(1)
        elif lista[i] == 'b':
            newdata_N.append(0)
        else:
            print("Erro na entrada de dados, digite os valores novamente")
            error = True
            break
    if error is True:
        error = False
        continue

    test = model.predict([newdata_N])
    if test == 1 or test == '1':
        print("X ganhou")
    else:
        print("X perdeu")

print("Fim")
