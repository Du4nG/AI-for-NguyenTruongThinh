from sklearn.preprocessing import StandardScaler
l1 = 40
l2 = 50
x_train = []
y_train = []

for t1 in np.linspace(-(2 * np.pi), 2 * np.pi, 500):
  for t2 in np.linspace(-(2 * np.pi), 2 * np.pi, 500):
    x = l1*m.cos(t1) + l2*m.cos(t1+t2)
    y = l1*m.sin(t1) + l2*m.sin(t1+t2)
    x_train.append(np.array([x,y]))
    y_train.append(np.array([t1,t2]))

scaler = StandardScaler()
x_train = np.array(scaler.fit_transform(x_train))
y_train = np.array(y_train)
x_train, y_train = shuffle(x_train, y_train)

model2 = Sequential()
model2.add(Dense(256, activation='relu', input_shape = (2,)))
model2.add(Dense(256, activation='relu'))
model2.add(Dense(256, activation='relu'))
model2.add(Dense(2, activation='linear'))

model2.compile(loss='mae', optimizer = tf.optimizers.Adam(learning_rate=0.0001))

test = scaler.transform(np.array([[90,0]]))
t1 = model2.predict(test)[0][0]
t2 = model2.predict(test)[0][1]

x = l1*m.cos(t1) + l2*m.cos(t2+t1)
y = l1*m.sin(t1) + l2*m.sin(t2+t1)

print("X = 90, Y = 0, t1 = " + str(t1) + ", t2 = "+ str(t2))
print("Result: X = " + str(x) + ", Y = "+ str(y))
