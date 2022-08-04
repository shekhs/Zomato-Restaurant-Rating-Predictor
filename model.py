import pandas as pd

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("Zomato_df_cleaned.csv")
X = df.drop(columns="rate",axis=1)
y = df.rate

###
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.35,random_state=42)
###
from sklearn.ensemble import ExtraTreesRegressor
et_model = ExtraTreesRegressor(n_estimators=65)
et_model.fit(X_train,y_train)
y_pred = et_model.predict(X_test)
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))

import bz2
# ofile = bz2.BZ2File("BinaryData",'wb')
# pickle.dump(data,ofile)
# ofile.close()

#
# ofile = open("BinaryData",'wb')
# pickle.dump(data, ofile)
# ofile.close()




import pickle
pickle.dump(et_model,bz2.BZ2File("model.pkl",'wb'))
model = pickle.load(bz2.BZ2File('model.pkl','rb'))
print(y_pred)

