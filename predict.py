import joblib 
x = input("Enter the expreience (in yrs) : ")
x = float(x)
model = joblib.load("trained_model.h5")
y = model.predict([[x]])
print("Expected salary is : " + str(y[0]))
