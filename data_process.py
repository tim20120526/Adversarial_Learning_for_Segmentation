import glob
import cv2
import  numpy as np
import random
def load_data(x_path='/home/tim/My-data/ISIC_2017/img/', y_path='/home/tim/My-data/ISIC_2017/GTs/'):
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
    x_test = []
    y_test = []
    for item in glob.glob(x_path + '*'):
        for path in glob.glob(item + '/*'):
            x = []
            y = []
            for file_path in glob.glob(path + '/*.jpg'):
                #                print(item)
                #                print(path)
                #                print(file_path)
                print('file_path:{}'.format(file_path))
                name_label = file_path.split('\\')[-1][0:8]
                print('name_label:{}'.format(name_label))
                label_path = y_path + name_label + '.png'
                img = cv2.imread(file_path, 2)
                #img = cv2.resize(img, (128, 96))
                print(img.shape)
                label_img = cv2.imread(label_path, 2)

                x.append(img)
                y.append(label_img)
            num = 10
            index = [n for n in range(num)]
            #            random.shuffle(index)
            count_train = int(num * 6 / 10)  # 6
            count_valid = int(num * 8 / 10)  # 8
            for i in range(count_train):
                x_train.append(x[index[i]])
                y_train.append(y[index[i]])
            for i in range(count_train, count_valid):
                x_valid.append(x[index[i]])
                y_valid.append(y[index[i]])
            for i in range(count_valid, num):
                x_test.append(x[index[i]])
                y_test.append(y[index[i]])
            '''
            6:2:2
            '''
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_valid = np.array(x_valid)
    y_valid = np.array(y_valid)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    x_train = np.expand_dims(x_train, axis=3)
    #    y_train=np.expand_dims(y_train,axis=3)
    x_valid = np.expand_dims(x_valid, axis=3)
    #    y_valid=np.expand_dims(y_valid,axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    #    y_test=np.expand_dims(y_test,a1xis=3)
    print(x_train.shape)
    print(y_train.shape)
    print(x_valid.shape)
    print(y_valid.shape)
    print(x_test.shape)
    print(y_test.shape)
    return x_train / 255, y_train / 255, x_valid / 255, y_valid / 255, x_test / 255, y_test / 255

def generator(data, label, batch_size, num, train):
    if train:
        #            print("************")
        index = [n for n in range(num)]
        random.shuffle(index)

        for j in range(num // batch_size):
            #                print("************")
            x = data[index[j * batch_size:(j + 1) * batch_size]]
            y = label[index[j * batch_size:(j + 1) * batch_size]]
            #                print(x.shape)
            #                print(y.shape)
            yield np.array(x), np.array(y)
    else:
        for j in range(num // batch_size):
            #                print("************")
            x = data[j * batch_size:(j + 1) * batch_size]
            y = label[j * batch_size:(j + 1) * batch_size]
            if j == (num // batch_size) - 1:
                x = data[j * batch_size:]
                y = label[j * batch_size:]
            yield np.array(x), np.array(y)
