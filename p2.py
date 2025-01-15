import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensortflow.keras.layers import Dense
data=load_breast_cancer()
X=data.data
y=data.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
activation_function = 'sigmoid'
model = Sequential()
model.add(Dense(16, input_dim=X_train.shape[1], activation=activation_function))
model.add(Dense(8,activation=activation_function))
model.add(Dense(1,activation=activation_function))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=50,batch_size=16,verbose=1)
loss,accuracy=model.evaluate(X_test,y_test,verbose=0)
print(f"test accuracy:{accuracy:.2f}")
sample_index=52
sample_data=X_test[sample_index].reshape(1,-1)
predicted_prob=model.predict(sample_data)[0][0]
predicted_class=int(predicted_prob>0.5)
print(f"predicted probability:{predicted_prob:.2f}")
if predicted_class==1:
    print("the model predicts: no cancer")
else:
    print("the model predicts: cancer")
