import xgboost as xgb
import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def loadData():
    '''
    background: 0
    pore: 1
    '''
    background_dir = './backgroundimg'
    pore_dir = './poresimg'
    # 目录下全是图片,且分辨率为20*20


    bg_imgs = os.listdir(background_dir)
    num_bg_imgs = len(bg_imgs)
    background_data = np.zeros((num_bg_imgs,400), dtype=np.uint8)
    background_label = np.zeros((num_bg_imgs,), dtype=np.uint8)
    for index, img in enumerate(bg_imgs):
        im = cv2.imread(os.path.join(background_dir, img), cv2.IMREAD_GRAYSCALE)
        background_data[index] = im.ravel()

    p_imgs = os.listdir(pore_dir)
    num_pore_imgs = len(p_imgs)
    pore_data = np.zeros((num_pore_imgs,400), dtype=np.uint8)
    pore_label = np.ones((num_pore_imgs,), dtype=np.uint8)
    for index, img in enumerate(p_imgs):
        im = cv2.imread(os.path.join(pore_dir, img), cv2.IMREAD_GRAYSCALE)
        pore_data[index] = im.ravel()

    
    all_data = np.vstack((background_data, pore_data))
    all_label = np.concatenate((background_label, pore_label))

    return all_data, all_label


if __name__ == "__main__":
    x, y = loadData()
    print('loading data...')
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

    print('training...')
    model = xgb.XGBClassifier()
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)
    y_predict = [ round(val) for val in y_predict]
    print(accuracy_score(y_test,y_predict))

    fpr,tpr,threshold = roc_curve(y_test, y_predict, pos_label=1)
    roc_auc = auc(fpr,tpr)

    plt.figure()
    plt.title('roc')
    plt.plot(fpr,tpr,lw=2,label='roc')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.show()