with open('dung.pickle', 'rb') as f:
    (x_train, y_train) = pickle.load(f)

x_pre = x_train[101]
x_train = x_train[:150]
y_train = y_train[:150]
x_train = x_train.reshape(x_train.shape[0], -1)

x_train = x_train.astype('float32')
x_train /= 255

y_train = np_utils.to_categorical(y_train, 2)
x_train, y_train = shuffle(x_train, y_train)

model1 = Sequential()
model1.add(Dense(10, activation='relu', input_shape = (67500,)))
model1.add(Dense(10, activation='relu'))
model1.add(Dense(10, activation='relu'))
model1.add(Dense(2, activation='sigmoid'))
model1.compile(loss='binary_crossentropy', optimizer =Adam(), metrics=['acc'])

plt.imshow(cv2.cvtColor(x_pre, cv2.COLOR_BGR2RGB))
print(x_pre.shape)
img = x_pre.reshape(1,-1)
img = img.astype('float32')
img /= 255

plt.title("1: Yes, 0: No. Result: " + str(np.argmax(model1.predict(img))))
plt.imshow(cv2.cvtColor(x_pre, cv2.COLOR_BGR2RGB), cmap=plt.get_cmap('gray'))
