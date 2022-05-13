from sklearn.preprocessing import StandardScaler
l1 = 40
l2 = 50
l3 = 20
x_train = []
y_train = []

for t1 in np.linspace(-(2 * np.pi), 2 * np.pi, 100):
  for t2 in np.linspace(-(2 * np.pi), 2 * np.pi, 100):
    for t3 in np.linspace(-(2 * np.pi), 2 * np.pi, 100):
      x = l1*m.cos(t1) + l2*m.cos(t1+t2) + l3*m.cos(t1+t2+t3)
      y = l1*m.sin(t1) + l2*m.sin(t1+t2) + l3*m.sin(t1+t2+t3)
      beta = (t1 + t2 + t3)*180/3.14
      x_train.append(np.array([x,y,beta]))
      y_train.append(np.array([t1,t2,t3]))

scaler = StandardScaler()
x_train = np.array(scaler.fit_transform(x_train))
y_train = np.array(y_train)
x_train, y_train = shuffle(x_train, y_train)

model3 = Sequential()
model3.add(Dense(256, activation='relu', input_shape = (3,)))
model3.add(Dense(256, activation='relu'))
model3.add(Dense(3, activation='linear'))
model3.compile(loss='mae', optimizer =tf.optimizers.Adam(learning_rate=0.0001))

test = scaler.transform(np.array([[75,0,45]]))
t1 = model3.predict(test)[0][0]
t2 = model3.predict(test)[0][1]
t3 = model3.predict(test)[0][2]

x = l1*m.cos(t1) + l2*m.cos(t1+t2) + l3*m.cos(t1+t2+t3)
y = l1*m.sin(t1) + l2*m.sin(t1+t2) + l3*m.sin(t1+t2+t3)
beta = (t1 + t2 + t3)*180/3.14

print("X = 90, Y = 0, Beta = 45, t1 = " + str(t1) + ", t2 = "+ str(t2) + ", t3 = "+ str(t3))
print("x = " + str(x) + ", Y = "+ str(y)+ ", B = "+ str(beta))
