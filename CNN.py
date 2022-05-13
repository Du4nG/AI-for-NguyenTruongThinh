with open('data.pickle', 'rb') as f:
    (x_train, y_train) = pickle.load(f)

x_pre_1 = x_train[101]
x_pre_2 = x_train[201]
x_pre_3 = x_train[301]

x_train = x_train.astype('float32')
x_train /= 255

y_train = np_utils.to_categorical(y_train, 4)
x_train, y_train = shuffle(x_train, y_train)

model4 = Sequential()
model4.add(Conv2D(32, (3,3), activation='relu',kernel_initializer='he_uniform', padding ='same', input_shape = (150,150,3)))
model4.add(Conv2D(32, (3,3), activation='relu',kernel_initializer='he_uniform', padding ='same'))
model4.add(MaxPooling2D(2,2))

model4.add(Conv2D(64, (3,3), activation='relu',kernel_initializer='he_uniform', padding ='same'))
model4.add(Conv2D(64, (3,3), activation='relu',kernel_initializer='he_uniform', padding ='same'))
model4.add(MaxPooling2D(2,2))

model4.add(Conv2D(128, (3,3), activation='relu',kernel_initializer='he_uniform', padding ='same'))
model4.add(Conv2D(128, (3,3), activation='relu',kernel_initializer='he_uniform', padding ='same'))
model4.add(MaxPooling2D(2,2))

model4.add(Flatten())
model4.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model4.add(Dense(4, activation='softmax'))
model4.summary()

opt = Adam(lr = 0.001)
model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['acc'])
his = model.fit(x_train, y_train, epochs = 15, batch_size = 64, validation_split = 0.2)

label = ['Ngọc', 'Dũng', 'Mập']

plt.title("Predicted model:  " + label[np.argmax(model4.predict(x_pre_1.reshape(1,150,150,3)))])
plt.imshow(cv2.cvtColor(x_pre_1, cv2.COLOR_BGR2RGB), cmap=plt.get_cmap('gray'))  
# 3 thằng thì thay x_pre 
