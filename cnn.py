import cv2
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
TRAIN_DIR = 'augmanted'
TEST_DIR = 'Test'
IMG_SIZE = 50
LR = 0.001
MODEL_NAME = 'sports-cnn'
def create_label(image_name,image_data):
    """ Create an one-hot encoded vector from image name """
    word_label = image_name.split('_')[0]
    print(word_label)
    if word_label == 'Basketball':
        return np.array([1,0,0,0,0,0])
    elif word_label == 'Football':
        return np.array([0,1,0,0,0,0])
    elif word_label == 'Rowing':
        return np.array([0,0,1,0,0,0])
    elif word_label == 'Swimming':
        return np.array([0,0,0,1,0,0])
    elif word_label == 'Tennis':
        return np.array([0,0,0,0,1,0])
    elif word_label == 'Yoga':
        return np.array([0,0,0,0,0,1])
def augmantation():
    TRAIN_DIR = 'Train'
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    IMG_SIZE = 150
    for img in tqdm(os.listdir(TRAIN_DIR)):
        word_label = img.split('_')[0]
        path = os.path.join(TRAIN_DIR, img)
        image = cv2.imread(path, 1)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(image)
        # img=load_img('flo/flower_1.jpg')
        # x=img_to_array(img)
        image = image.reshape((1,) + image.shape)
        i = 0


        for batch in datagen.flow(image, batch_size=1, save_to_dir='augmanted', save_prefix=f'{word_label}_', save_format='.jpg'):
            i += 1
            if i > 5:
                break
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        path = os.path.join(TRAIN_DIR, img)
        img_data = cv2.imread(path,1)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img_data), create_label(img,img_data)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def testing_dataset(model):
    testing_data=[]
    # ['Basketball','football','Rowing','Swimming','Tennis','Yoga']
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_data = cv2.imread(path, 1)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        img_data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
        prediction = model.predict([img_data])[0]
        prediction = np.argmax(prediction)
        predicted_sport_name=""
        # ['Basketball','football','Rowing','Swimming','Tennis','Yoga']
        # if(prediction==0):predicted_sport_name="Basketball"
        # elif(prediction==1):predicted_sport_name="Football"
        # elif (prediction == 2):
        #     predicted_sport_name = "Rowing"
        # elif (prediction == 3):
        #     predicted_sport_name = "Swimming"
        # elif (prediction == 4):
        #     predicted_sport_name = "Tennis"
        # elif (prediction == 5):
        #     predicted_sport_name = "Yoga"
        # word_label = img.split('.')[0]
        # print(word_label)
        testing_data.append([img,prediction])
    # shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    print('before')
    np.savetxt('output_test.csv',testing_data, delimiter=',',fmt='%s')
    print('after')
    # return testing_data
if(len(os.listdir(TRAIN_DIR))==0):
    augmantation()

if (os.path.exists('train_data.npy')): # If you have already created the dataset:
    train_data =np.load('train_data.npy',allow_pickle=True)
    #train_data = create_train_data()
else: # If dataset is not created:
    train_data = create_train_data()

#print(train_data.shape)
#
# if (os.path.exists('test_data.npy')):
#     test_data =np.load('test_data.npy',allow_pickle=True)
# else:
#     test_data = test_data()

#print(train_data.shape)
#print(test_data.shape)
train = train_data
# test = test_data
X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y_train = [i[1] for i in train]

# X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
# y_test = [i[1] for i in test]

tf.compat.v1.reset_default_graph()
conv_input = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')
conv1 = conv_2d(conv_input, 32, 5, activation='relu')
pool1 = max_pool_2d(conv1, 5)

conv2 = conv_2d(pool1, 64, 5, activation='relu')
pool2 = max_pool_2d(conv2, 5)

conv3 = conv_2d(pool2, 128, 5, activation='relu')
pool3 = max_pool_2d(conv3, 5)

conv4 = conv_2d(pool3, 64, 5, activation='relu')
pool4 = max_pool_2d(conv4, 5)

conv5 = conv_2d(pool4, 32, 5, activation='relu')
pool5 = max_pool_2d(conv5, 5)
conv6 = conv_2d(pool5, 32, 5, activation='relu')
pool6 = max_pool_2d(conv6, 5)
conv7 = conv_2d(pool6, 32, 5, activation='relu')
pool7 = max_pool_2d(conv7, 5)
fully_layer = fully_connected(pool5, 1024, activation='relu')
fully_layer = dropout(fully_layer, 0.5)
cnn_layers = fully_connected(fully_layer, 6, activation='softmax')

cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)
print (X_train.shape)

if (os.path.exists('model.tfl.meta')):
    model.load('./model.tfl')
else:
    mo=model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10,
          # validation_set=({'input': X_test}, {'targets': y_test}),
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

    model.save('model.tfl')

testing_dataset(model)



# ['Basketball','football','Rowing','Swimming','Tennis','Yoga']
#
# img = cv2.imread('6.jpg',1)
# img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# test_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
# test_img = test_img.reshape(IMG_SIZE, IMG_SIZE, 3)
# prediction = model.predict([test_img])[0]
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111)
# ax.imshow(img,cmap='gray')
# prediction=np.argmax(prediction)
# # ['Basketball','football','Rowing','Swimming','Tennis','Yoga']
# print (prediction)
# # print(f"basketball: {prediction[0]}, football: {prediction[1]},rowing: {prediction[2]},swimming:{prediction[3]},tennis: {prediction[4]},yoga: {prediction[5]}")
# plt.show()
#
