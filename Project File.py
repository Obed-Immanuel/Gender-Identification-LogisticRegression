import cv2
import numpy as np
#from matplotlib import pyplot as plt

import os
import glob

#Global input data and expected output
#Initilized
n = 2500
X = np.empty((0, n), int)
y = np.empty((0), int)

#Function to make a dataset from raw images in a directory
#Parameters : m-number of training examples
#           : n-number of Features
#           # Directory of Men and Women images
def makeDataset(X, y, M_imgDir, W_imgDir):
    dataPath_M = os.path.join(M_imgDir,'*g')
    files_M = glob.glob(dataPath_M)


    dataPath_W = os.path.join(W_imgDir,'*g')
    files_W = glob.glob(dataPath_W)


    #img = cv2.imread('TotalDataset/man/face_0.jpg', 0)
    #print(img)
    #plt.imshow(img, cmap = 'gray'), plt.axis('off')
    #plt.show()
        
    
    for F in files_M:
        img = cv2.imread(F, 0)
        img_50x50 = cv2.resize(img, (50,50))
        img_flat = img_50x50.reshape((1, n))
        X = np.append(X, img_flat, axis = 0)
        y = np.append(y, 1)
        
    for F in files_W:
        img = cv2.imread(F, 0)
        img_50x50 = cv2.resize(img, (50,50))
        img_flat = img_50x50.reshape((1, n))
        X = np.append(X, img_flat, axis = 0)
        y = np.append(y, 0)
            
    return X,y
        #X[10].reshape((50,50))
        #plt.imshow(X[1130].reshape((50,50)), cmap = 'gray'), plt.axis('off')

#Assigning Directory
M_imgDir = 'TotalDataset/man'
W_imgDir = 'TotalDataset/woman'

#Calling the function with m and n values, Directory locations of Men and Women images
X, y = makeDataset(X, y, M_imgDir, W_imgDir)

#Function to plot array of random images
# def displayData(X, num):
#     row, column = X.shape
#     perm = np.random.permutation(row)
#     X_perm = X[perm[:num], :]
    
#     for i in range(0,num):
#         img = X_perm[i].reshape((50,50))
#         plt.imshow(img, cmap = 'gray')
    

#displayData(X, 1)

#Some important variables
m, n = X.shape

# def featureNormalize(X):
#     m, n = X.shape
#     mu = np.zeros((1,n))
#     sigma = np.zeros((1, n))
#     mu = np.mean(X)
#     sigma = np.std(X);
#     X_norm = (X - mu) / sigma
#     return X_norm

#X = featureNormalize(X)
#Adding Bias term
X = np.append(np.ones((m, 1)), X, axis = 1)


#Split dataset int train set, test set, validation set
from sklearn.model_selection import train_test_split
X_train, X_t, y_train, y_t = train_test_split(X, y, test_size = 0.4)
X_test, X_cv, y_test, y_cv = train_test_split(X_t, y_t, test_size = 0.5)

#Some important variables
#m_train = X_train.shape[0]


#Initilizing parameters
init_theta = np.zeros(n+1)

#Sigmoid FunctionS
def sigmoid(z):
    return 1/(1+np.exp(-z))


    
#Logistic Regression
#Cost Function with regularisation 
#Outputs Cost
def cost_function(theta, X, y, lam):
    m, n = X.shape
    theta = theta.reshape((n,1))
    y = y.reshape((m,1))
    hypo = sigmoid(X @  theta)
#    J = (1/m) * (-y.T @ np.log(hypo) - (1-y).T @ np.log(1-hypo))    #Without Regularisation
#    J = ((-1/m) * (y.T @ mylog(hypo) + (1-y).T @ mylog(1-hypo)))
    term1 = hypo
    term1[hypo == 0] = 1
    term2 = 1-hypo
    term2[hypo == 1] = 1
    J = ((-1/m) * (y.T @ np.log(term1) + (1-y).T @ np.log(term2)))
#    J = ((-1/m) * sum(xlog1py(y, expit(hypo)) + xlog1py(1-y, -expit(1-hypo))))
    reg = (lam / (2*m) * sum(theta[1:]**2))
    return J+reg

cost_function(init_theta, X_train, y_train, 0)

#Outputs Gradient
def cost_grad(theta, X, y, lam):
    m, n = X.shape
    theta = theta.reshape((n,1))
    y = y.reshape((m,1))
    hypo = sigmoid(np.dot(X, theta))
    grad = np.zeros((n,1))
 #   grad = (1/m) * (X.T @ (hypo - y))   #Without Regularisation
    grad[0] = 1/m * sum(hypo - y)
    grad[1:] = (1/m * X[:,1:].T @ (hypo - y)) + lam/m * theta[1:]
    return  grad.flatten()

import scipy.optimize as opt

#Function to predict
def predict(X, theta):
    h = (sigmoid(X @ theta))
    p = h >= 0.5
    return h, p

#Bias/variance as a function of regularisation parameter
#def regularisationCurve(X, y, t):
#    lam = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
#    error_cv = np.zeros(10)
#    for i in range(len(lam)):
#        theta = opt.fmin_cg(f = cost_function, x0 = init_theta, fprime = cost_grad, args=(X_train, y_train, lam[i]))
#        y_hypo_cv, y_pred_cv = predict(X_cv, theta)
#        e = np.mean(y_pred_cv == y_cv) * 100
#        error_cv[i] = e
#    plt.plot(lam, error_cv)

#regularisationCurve(X_cv, y_cv, init_theta)

#Training the model (Fitting Theta using gradient descent)
optimal_theta = opt.fmin_cg(f = cost_function, x0 = init_theta, fprime = cost_grad, args=(X_train, y_train, 1))
#res = opt.minimize(fun = cost_function, x0 = init_theta, args = (X_train, y_train, 1), method = 'CG', jac = cost_grad)
#optimal_theta = res.x

#y_pred in test case
y_hypo_test, y_pred_test = predict(X_test, optimal_theta.T)

#Error Analysis
e = np.mean(y_pred_test == y_test) * 100
print("Expected Accuracy is %f" %e)

#Learning Curve
# def learningCurve(X, y, X_val, y_val, lam):
#     m, n = X.shape
#     error_train = np.zeros(int(m/200) +15)
#     error_val = np.zeros(int(m/200) + 15)
#     m_train = np.zeros(int(m/200) + 15)
#     theta = np.zeros(n)
#     for i in range(1, m, 200):
#         init_theta = np.zeros(n)
#         theta = 0
#         theta = opt.fmin_cg(f = cost_function, x0 = init_theta, fprime = cost_grad, args=(X_train[:i+1, :], y_train[:i+1], lam))
#         error_train[int(i/200)] = cost_function(theta, X_train, y_train, 0)
#         error_val[int(i/200)] = cost_function(theta, X_val, y_val, 0)
#         m_train[int(i/200)] = i
#     return (error_train, error_val, m_train)

#error_train, error_cv, m_train = learningCurve(X_train, y_train, X_cv, y_cv, 1)

#for i in range(1, m, 200):
#    m_train[int(i/200)] = i
    
#plt.plot(m_train, error_train, 'r', label = 'Training set')
#plt.plot(m_train, error_cv, 'b', label = 'Validation set')
#plt.xlabel("Number of Examples")
#plt.ylabel("Error")
#plt.legend()

#predict new values form web cam
def capture():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Test")

    while True:
        flag = 0
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab Frame")
            break
        cv2.imshow("test", frame)
        k = cv2.waitKey(1)
        
        if(k % 256 == 27):
            flag = -1
            img_flat = np.zeros((1,n+1))
            break
    
        elif(k % 256 == 32):
            #space pressed
            img_name = "Test_img_0.png"
            cv2.imwrite(img_name, frame)
            #print("{} Written !".format(img_name))
            break
        
    cam.release()
    cv2.destroyAllWindows()
    
    if(flag != -1):
        img = cv2.imread(img_name, 0)
        #Detucting Face
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))
        if(len(faces)==0): 
            flag = 0
            img_flat = np.zeros((1,n+1))
        else: 
            flag = 1
            for(x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face = img[y:y+h, x:x+w]
                break
            #cv2.imwrite('FaceOnly.jpg', img)
            img_50x50 = cv2.resize(face, (50,50))
            #plt.imshow(img_50x50.reshape((50,50)), cmap = 'gray')
            img_flat = img_50x50.flatten()
            img_flat = np.append(1, img_flat)
    return flag, img_flat
    
import tkintertable
def msgBox(y, val, flag):
    if flag == 0 : msg = "No faces Detucted"
    elif(y == 1):
        msg = "Male with {}% Probability".format(val)
    else:
        msg = "Female with {}% Probability".format(100-val)
    tkintertable.messagebox.showinfo(title = "Prediction Results", message = msg)
    #tkintertable.messagebox.


#predict for captured image
while (True):
    flag, X_cap = capture()
    if(flag == -1): break
    elif(flag == 1):
        hypo, prediction = predict(X_cap, optimal_theta)
    else:
        hypo, prediction = 0,0
    msgBox(prediction, hypo * 100, flag)
#    print(sigmoid(X_cap @ optimal_theta) * 100)
#    if(prediction == 1):
#        print("Male")
#    else:
#        print("Female")