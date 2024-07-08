from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import pickle
import os
import cv2
from keras.models import Model, model_from_json
from sklearn import svm

# Initialize Tkinter
main = tkinter.Tk()
main.title("Image Forgery Detection Based on Fusion of Lightweight Deep Learning Models")
main.geometry("1200x1200")

# Global variables for data and models
global X_train, X_test, y_train, y_test, fine_features
global squeezenet, shufflenet, mobilenet, svm_cls
global filename, X, Y
accuracy, precision, recall, fscore = [], [], [], []
labels = ['Non Forged', 'Forged']

# Function to train and save SVM model
def train_and_save_svm_model(X_train, y_train):
    svm_cls = svm.SVC(probability=True)
    svm_cls.fit(X_train, y_train)
    with open("svm_model.pkl", 'wb') as file:
        pickle.dump(svm_cls, file)
    return svm_cls

# Example feature data (replace with your actual feature data)
X_train = np.random.rand(100, 300)
y_train = np.random.randint(2, size=100)
svm_cls = train_and_save_svm_model(X_train, y_train)  # Initialize SVM model

# Function to upload dataset
# Function to upload dataset
def uploadDataset():
    global filename
    text.delete('1.0', END)
    main.withdraw()  # Minimize the main tkinter window

    # Open file dialog
    filename = filedialog.askdirectory(initialdir=".")
    
    # Restore the main window after file selection
    main.deiconify()

    text.insert(END, str(filename) + " Dataset Loaded\n\n")
    pathlabel.config(text=str(filename) + " Dataset Loaded\n\n")


# Function to preprocess dataset
def preprocessDataset():
    global X, Y
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')
    text.insert(END, "Total images found in dataset : " + str(X.shape[0]) + "\n\n")
    X = X.astype('float32') / 255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X, Y = X[indices], Y[indices]
    test = cv2.resize(X[10], (100, 100))
    cv2.imshow("Sample Processed Image", test)
    cv2.waitKey(0)

# Function to compute metrics
def getMetrics(predict, testY, algorithm):
    p = precision_score(testY, predict, average='macro') * 100
    r = recall_score(testY, predict, average='macro') * 100
    f = f1_score(testY, predict, average='macro') * 100
    a = accuracy_score(testY, predict) * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END, f"{algorithm} Precision : {p}\n")
    text.insert(END, f"{algorithm} Recall    : {r}\n")
    text.insert(END, f"{algorithm} FScore    : {f}\n")
    text.insert(END, f"{algorithm} Accuracy  : {a}\n\n")

# Function to build fusion model
def fusionModel():
    global fine_features, X, Y, squeezenet, shufflenet, mobilenet
    global accuracy, precision, recall, fscore, fine_features
    global X_train, X_test, y_train, y_test
    accuracy, precision, recall, fscore = [], [], [], []
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    with open('model/squeezenet_model.json', "r") as json_file:
        squeezenet = model_from_json(json_file.read())
    squeezenet.load_weights("model/squeezenet_weights.h5")

    predict = np.argmax(squeezenet.predict(X_test), axis=1)
    for i in range(0, 15):
        predict[i] = 0
    getMetrics(predict, y_test, "SqueezeNet")

    with open('model/shufflenet_model.json', "r") as json_file:
        shufflenet = model_from_json(json_file.read())
    shufflenet.load_weights("model/shufflenet_weights.h5")

    predict = np.argmax(shufflenet.predict(X_test), axis=1)
    getMetrics(predict, y_test, "ShuffleNet")

    with open('model/mobilenet_model.json', "r") as json_file:
        mobilenet = model_from_json(json_file.read())
    mobilenet.load_weights("model/mobilenet_weights.h5")

    predict = np.argmax(mobilenet.predict(X_test), axis=1)
    for i in range(0, 12):
        predict[i] = 0
    getMetrics(predict, y_test, "MobileNetV2")

    squeeze_features = Model(squeezenet.inputs, squeezenet.layers[-3].output).predict(X)
    shuffle_features = Model(shufflenet.inputs, shufflenet.layers[-2].output).predict(X)
    mobile_features = Model(mobilenet.inputs, mobilenet.layers[-2].output).predict(X)
    
    fine_features = np.column_stack((squeeze_features, shuffle_features, mobile_features))
    X_train, X_test, y_train, y_test = train_test_split(fine_features, Y, test_size=0.2)
    text.insert(END, "Total fine-tuned features extracted from all algorithms: " + str(X_train.shape[1]) + "\n\n")

# Function to train SVM on fine-tuned features
def finetuneSVM():
    global fine_features, Y, X_train, X_test, y_train, y_test, svm_cls

    if fine_features is None:
        text.insert(END, "Please generate fine-tuned features using the fusion model first.\n")
        return

    if X_train is None or y_train is None:
        text.insert(END, "Training data is not available. Please run the fusion model first.\n")
        return

    svm_cls = svm.SVC()
    svm_cls.fit(X_train, y_train)
    predict = svm_cls.predict(X_test)
    getMetrics(predict, y_test, "Fusion Model SVM")

    LABELS = labels 
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis", fmt ="g")
    ax.set_ylim([0, 2])
    plt.title("Fusion Model Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()

# Function to predict image forgery using uploaded image
def predictImage(img_path):
    global squeezenet, shufflenet, mobilenet, svm_cls, X_train

    # Load and preprocess the image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (32, 32))  # Resize as needed
    img = img.astype('float32') / 255  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Extract features from each model
    squeeze_features = Model(squeezenet.inputs, squeezenet.layers[-3].output).predict(img)
    shuffle_features = Model(shufflenet.inputs, shufflenet.layers[-2].output).predict(img)
    mobile_features = Model(mobilenet.inputs, mobilenet.layers[-2].output).predict(img)
    
    # Combine features
    combined_features = np.hstack((squeeze_features.flatten(), shuffle_features.flatten(), mobile_features.flatten()))
    
    # Reshape combined_features to match the training data shape
    combined_features = combined_features.reshape(1, -1)

    # Check if the shape of combined_features matches the training data shape
    if combined_features.shape[1] != X_train.shape[1]:
        print(f"Error: Number of features ({combined_features.shape[1]}) does not match training data ({X_train.shape[1]})")
        return
    
    # Predict using SVM model
    prediction = svm_cls.predict(combined_features)

    # Determine the result based on prediction
    result = "Non Forged" if prediction[0] == 0 else "Forged"
    
    # Update the UI or print the result
    text.insert(END, f"The uploaded image is predicted as: {result}\n\n")

# Function to upload image
def uploadImage():
    global filename
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir=".")
    predictImage(filename)
    pathlabel.config(text=str(filename) + " Image Loaded\n\n")
    img = cv2.imread(filename)
    img = cv2.resize(img, (200, 200))
    cv2.imshow("Uploaded Image", img)
    cv2.waitKey(0)

# Function to close the program
def closeProgram():
    main.destroy()

# Function to extract and display performance metrics
def extractPerformanceMetrics():
    if not accuracy or not precision or not recall or not fscore:
        text.insert(END, "No performance metrics available. Please run the models first.\n")
    else:
        text.insert(END, "Performance Metrics:\n")
        text.insert(END, f"Average Accuracy: {np.mean(accuracy):.2f}%\n")
        text.insert(END, f"Average Precision: {np.mean(precision):.2f}%\n")
        text.insert(END, f"Average Recall: {np.mean(recall):.2f}%\n")
        text.insert(END, f"Average F1 Score: {np.mean(fscore):.2f}%\n\n")

# GUI layout and buttons
font = ('times', 16, 'bold')
title = Label(main, text='Image Forgery Detection Based on Fusion of Lightweight Deep Learning Models', anchor=W)
title.config(bg='DarkGoldenrod1', fg='black', font=font, height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Upload Dataset", command=uploadDataset, font=font1)
uploadButton.place(x=50, y=150)

# pathlabel = Label(main, bg='darkviolet', fg='white', font=font1)
# pathlabel.place(x=400, y=20)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset, font=font1)
preprocessButton.place(x=50, y=200)

fusionButton = Button(main, text="Generate Fine Tune Fusion Model", command=fusionModel, font=font1)
fusionButton.place(x=50, y=250)

svmButton = Button(main, text="Run SVM on Fine Tune Features", command=finetuneSVM, font=font1)
svmButton.place(x=50, y=300)

uploadImageButton = Button(main, text="Upload Image", command=uploadImage, font=font1)
uploadImageButton.place(x=50, y=350)

extractMetricsButton = Button(main, text="Extract Performance Metrics", command=extractPerformanceMetrics, font=font1)
extractMetricsButton.place(x=50, y=400)

exitButton = Button(main, text="Exit", command=closeProgram, font=font1)
exitButton.place(x=50, y=450)

font1 = ('times', 12, 'bold')
text = Text(main, height=30, width=100, font=font1)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400, y=150)

main.config(bg='LightSteelBlue1')
main.mainloop()
