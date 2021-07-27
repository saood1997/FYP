from django.shortcuts import render, redirect
import os
import json
import glob
import copy
import shutil
from PIL import Image as pImage
from django.conf import settings
from .forms import VideoUploadForm
import keras
from keras.models import  Model
from keras.layers import Dense, Conv2D , Flatten
from keras.initializers import RandomNormal
from keras.layers import Input, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D,MaxPooling2D, Add, AveragePooling2D
from keras.optimizers import Adam
import cv2 as cv
from glob import glob
from numpy import savez_compressed, load
from sklearn.metrics import confusion_matrix
import cv2 as cv
import numpy as np
import time
import random
import matplotlib.pylab as plt
from PIL import Image
import matplotlib.patches as patches
import copy
from keras.utils import to_categorical

index_template_name = 'index.html'
predict_template_name = 'predict.html'




class Model:
    def __init__(self) -> None:
        self.model = None

    def identityBlock(self, X, filters):
        #X is batch of images
        FX = X

        #First layer
        FX = Conv2D(filters = filters[0],  strides = 1, kernel_size = 1,  kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), padding = 'valid')(FX)
        #maintain the mean close to zero and std derivattion close to one
        FX = BatchNormalization(epsilon=0.00005, trainable=True)(FX)
        #max(0,x)
        FX = Activation('relu')(FX)
        
        #Second layer 
        FX = Conv2D(filters = filters[1],  strides = 1, kernel_size = 3,kernel_initializer = RandomNormal(mean=0.0, stddev=0.02), padding = 'same')(FX)
        FX = BatchNormalization(epsilon=0.00005, trainable=True)(FX)
        FX = Activation('relu')(FX)

        #Third Layer
        FX = Conv2D(filters = filters[2], strides = 1, kernel_size = 1, kernel_initializer = RandomNormal(mean=0.0, stddev=0.02), padding = 'valid')(FX)
        FX = BatchNormalization(epsilon=0.00005, trainable=True)(FX)

        #Add the FX and X 
        FX = Add()([FX, X])
        FX = Activation('relu')(FX)
        return FX

    def convolutionalBlock(self, X, filters):
        #X is batch of images
        FX = X
        #Extra convolution layer
        X = Conv2D(filters = filters[2], strides = 2, kernel_size = 1, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), padding = 'valid')(X)
        X = BatchNormalization(epsilon=0.00005, trainable=True)(X)
        
        #First layer
        FX = Conv2D(filters = filters[0], strides = 2, kernel_size = 1, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), padding = 'valid')(FX)
        FX = BatchNormalization(epsilon=0.00005, trainable=True)(FX)
        FX = Activation('relu')(FX)

        #Second Layer
        FX = Conv2D(filters = filters[1], strides = 1, kernel_size = 3, kernel_initializer = RandomNormal(mean=0.0, stddev=0.02), padding = 'same')(FX)
        FX = BatchNormalization(epsilon=0.00005, trainable=True)(FX)
        FX = Activation('relu')(FX)

        #Third Layer
        FX = Conv2D(filters = filters[2], strides = 1, kernel_size = 1, kernel_initializer = RandomNormal(mean=0.0, stddev=0.02), padding = 'valid')(FX)
        FX = BatchNormalization(epsilon=0.00005, trainable=True)(FX)

        #Add the FX and X
        FX = Add()([FX, X])
        FX = Activation('relu')(FX)
        return FX
    
    def resNet50(self, input_image):
        #input_image is the shape of image   
        input_shape = Input(input_image)
        #Zero Padding
        FX = ZeroPadding2D((3, 3))(input_shape)
        #Layer
        FX = Conv2D(64, kernel_size = 7, strides = 2, kernel_initializer = RandomNormal(mean=0.0, stddev=0.02))(FX)
        FX = BatchNormalization(epsilon=0.00005, trainable=True)(FX)
        FX = Activation('relu')(FX)
        FX = MaxPooling2D((3, 3), strides=2)(FX)
        
        #1st stage
        filters = [64, 64, 256]
        FX = self.convolutionalBlock(FX, filters)
        FX = self.identityBlock(FX, filters)
        FX = self.identityBlock(FX, filters)

        #2nd stage
        filters = [128, 128, 512]
        FX = self.convolutionalBlock(FX, filters)
        FX = self.identityBlock(FX, filters)
        FX = self.identityBlock(FX, filters)
        FX = self.identityBlock(FX, filters)

        #3rd stage
        filters = [256, 256, 1024]
        FX = self.convolutionalBlock(FX, filters)
        FX = self.identityBlock(FX, filters)
        FX = self.identityBlock(FX, filters)
        FX = self.identityBlock(FX, filters)
        FX = self.identityBlock(FX, filters)
        FX = self.identityBlock(FX, filters)

        #4th stage
        filters = [512, 512, 2048]
        FX = self.convolutionalBlock(FX, filters)
        FX = self.identityBlock(FX, filters)
        FX = self.identityBlock(FX, filters)
        #Avg Pooling
        FX = AveragePooling2D((2, 2))(FX)
        
        FX = Flatten()(FX)
        
        #Output Layer
        Fy = Dense(2, kernel_initializer = RandomNormal(mean=0.0, stddev=0.02), activation='sigmoid')(FX)
        model = keras.Model(inputs=input_shape, outputs=Fy, name='ResNet50')
        return model

    def loadWeights(self):
        save_load_path = 'weights/weight_81_6.h5' #change the path
        #load the previous weights
        self.model.load_weights(save_load_path)


    def loadAndCompileModel(self):
        #image shape
        img_shape = (128,128,3)
        #optimizer
        opt = Adam(learning_rate=1e-5)
        self.model = self.resNet50(img_shape)
        self.loadWeights()
        #compile the model
        self.model.compile(optimizer=opt, metrics=['accuracy'], loss=keras.losses.binary_crossentropy)

    def getModel(self):
        self.loadAndCompileModel()
        return self.model

class preProcessor:
    def __init__(self) -> None:
        self.faces_list = []
        self.frame_list = []
    
    def cropFaces(self,image):
        #image into grayscale
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        faceCascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
        #detect the face and return the rectangle for faces
        face_locations = faceCascade.detectMultiScale(
            gray_image,
            scaleFactor=1.3,
            minSize=(128, 128),
            minNeighbors=3
        )    
        faces = []
        flag = True
        for (x1, y1, x2, y2) in face_locations:
            if flag:
                flag = False
            try:
                crop_image = image[y1-10:y1+y2+10, x1-10:x1+x2+10]
            except:
                crop_image = image[y1:y1+y2, x1:x1+x2]
            faces.append(crop_image)
        return faces
    
    def getFrame(self,vid_cap,sec):
        #set the frame of second
        vid_cap.set(cv.CAP_PROP_POS_MSEC,1000*sec)
        success , image = vid_cap.read()
        if success:
            self.frame_list.append(image)
            faces = self.cropFaces(image)
            for face in faces:
                #with resize of image
                self.faces_list.append(cv.resize(face,(128,128)))    
        return success
        
    def videoToFrames(self,video_path):
        vid_cap = cv.VideoCapture(video_path)
        second = 0
        frameRate = 1 # Change the number of frame rate
        flag = self.getFrame(vid_cap,second)
        while flag:
            second = second + frameRate
            second = round(second, 2)
            flag = self.getFrame(vid_cap,second)

    def setVideo(self,video_path):
        self.videoToFrames(video_path)

    def getFaces(self):
        return self.faces_list
    
    def getFrames(self):
        return self.frame_list

class evaluator:
    def __init__(self,y_true,pred) -> None:
        self.true_negative = None
        self.false_positive = None
        self.false_negative = None
        self.true_positive = None
        
        
    def setMatrix(self, y_true, pred):
        self.true_negative, self.false_positive, self.false_negative, self.true_positive = confusion_matrix(y_true,pred).ravel()


    def getAccuracy(self):
        accuracy = ((self.true_positive+self.true_negative)/(self.true_negative+self.false_positive+self.false_negative+self.true_positive))
        return accuracy

    def getPrecision(self):
        precision = ((self.true_positive)/(self.true_positive+self.false_positive))
        return precision
    
    def getRecall(self):
        recall = ((self.true_positive)/(self.true_positive+self.false_negative))
        return recall

    def getF1_Score(self):
        f1_score = ((2*(self.getPrecision()*self.getRecall()))/(self.getPrecision()+self.getRecall()))
        return f1_score

class trainer:

    def __init__(self) -> None:
        self.model = None

    
    def loadNumpyData(self, out_data, out_label, count):
        #load the traning data
        load_numpy_array = load('numpyDataset/training/training_numpy_data_'+str(count)+'.npz',allow_pickle=True)
        images = []
        labels = []
        c = 0
        for array in load_numpy_array['arr_0']:
            try:
                #resize the images
                images.append(cv.resize(array[0],(128,128)))
                labels.append(array[1])
            except:
                c += 1 
        return np.asarray(images),np.asarray(labels)

    #important
    def getBatches(self, data, labels,batch_size):
        batches = []
        batches_labels = []
        for i in range(int(data.shape[0] // batch_size)+1):
            batch = data[i * batch_size:(i + 1) * batch_size]
            batch_labels = labels[i * batch_size:(i + 1) * batch_size]
            images = []
            for img in batch:
                image = Image.fromarray(img)
                # Flip some images horizontally
                if random.choice([True,False]):
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                images.append(np.asarray(image))
            batch = np.asarray(images)
            if i < int(data.shape[0] // batch_size):
                # Normalize images from 0 / 255 to -1 / 1
                normalized_batch = (batch / 127.5) - 1.0
                batches.append(normalized_batch)
                batches_labels.append(to_categorical(batch_labels, 2))
            else:
                batches.append(batch)
                batches_labels.append(batch_labels)
        return batches, batches_labels

    def saveWeights(self, epc):
        save_load_path = 'modelWeights/'
        #save the weight
        self.model.save_weights(save_load_path+'weight'+epc+'.h5')
    
    def savePlots(train_loss, train_acc, epc):
        save_load_path = 'plots/'
        plt.plot(train_loss,label='Train Loss', alpha=0.6)
        plt.title("Losses")
        plt.xlabel('batch #')
        plt.ylabel('loss value')
        plt.legend()
        plt.savefig(save_load_path+'loss'+epc+'.png')
        plt.close()
        plt.plot(train_acc, label='Train Accuracy', alpha=0.6)
        plt.title("Training Accuracy")
        plt.xlabel('batch #')
        plt.ylabel('accuracy value')
        plt.legend()
        plt.savefig(save_load_path+'accuracy'+epc+'.png')
        plt.close()

    def train(self, epochs, batch_size):
        start = 11
        for epoch in range(start,epochs+start):
            out_data = []
            out_label = []
            train_batch_accuracy = []
            train_batch_losses = []
            start_time = time.time()
            mini = 1
            for i in range(1,9):
                #load the images
                X_train, y_train = self.loadNumpyData(out_data,out_label,i)
                out_data = []
                out_label = []
                #image convert into batches
                batches,batch_labels = self.getBatches(X_train,y_train,batch_size)
                mini_epochs = len(batches)
                for mini_epoch in range(mini_epochs):
                    if len(batches[mini_epoch]) < batch_size:
                        for array in batches[mini_epoch]:
                            out_data.append(array)
                        for label in batch_labels[mini_epoch]:
                            out_label.append(label)
                        break
                    loss_acc = self.model.train_on_batch(batches[mini_epoch], batch_labels[mini_epoch])
                    train_batch_losses.append(loss_acc[0])
                    train_batch_accuracy.append(loss_acc[1])
                    print("Batch Train " + str(mini) + " in epoch " + str(epoch) + " with loss: " + str(loss_acc[0])+ " and accuracy: " + str(loss_acc[1])+  " finished in " + str(time.time() - start_time))
                    mini += 1
            print("Epoch " + str(epoch) + " finished in " + str(time.time() - start_time))
            self.saveWeights('_epoch_'+str(epoch+1))
            self.savePlots(train_batch_losses,train_batch_accuracy,'_epoch_'+str(epoch+1))

class predictor:
    def __init__(self) -> None:
        self.model = None
        self.faces = None
        self.pred = None

    def allowedVideo(self,filename):
        #check the video format
        allowed_video = ['mp4','gif','webm','avi','3gp','mkv']
        if (filename.split('.')[1] in allowed_video):
            return False
        else: 
            return True

    def loadVideo(self,path):
        self.extractFaces(path)

    def extractFaces(self,path):
        pre_inst = preProcessor()
        pre_inst.setVideo(path)
        self.faces = pre_inst.getFaces()
        self.frames = pre_inst.getFrames()
    
    def getFaces(self):
        return self.faces
    
    def getFrames(self):
        return self.frames

    def loadModel(self):
        inst_mod = Model()
        self.model = inst_mod.getModel()

    def predictFaces(self):
        self.loadModel()
        #predict the faces
        self.pred = np.argmax(self.model.predict(np.asarray(self.faces)), axis=-1)
        return self.pred
    
    def checkStatus(self):
        labels, label_counts = np.unique(self.pred, return_counts=True)
        if len(label_counts) == 1:
            if labels == 1:
                return True
            return False
        elif label_counts[0] > label_counts[1]:
            return False
        else:
            return True


pred_inst = predictor()

def index(request):
    if request.method == 'GET':
        video_upload_form = VideoUploadForm()
        if 'file_name' in request.session:
            del request.session['file_name']
        if 'preprocessed_images' in request.session:
            del request.session['preprocessed_images']
        if 'faces_cropped_images' in request.session:
            del request.session['faces_cropped_images']
        return render(request, index_template_name, {"form": video_upload_form})
    else:
        video_upload_form = VideoUploadForm(request.POST, request.FILES)
        if video_upload_form.is_valid():
            video_file = video_upload_form.cleaned_data['upload_video_file']
            video_file_ext = video_file.name.split('.')[-1]
            sequence_length = video_upload_form.cleaned_data['sequence_length']
        
            if pred_inst.allowedVideo(video_file.name):
                video_upload_form.add_error("upload_video_file","Only video are allowed ")
                return render(request, index_template_name, {"form": video_upload_form})
            saved_video_file = 'uploaded_file_'+str(int(time.time()))+"."+video_file_ext
            with open(os.path.join(settings.PROJECT_DIR, 'uploaded_videos', saved_video_file), 'wb') as vFile:
                shutil.copyfileobj(video_file, vFile)

            request.session['file_name'] = os.path.join(settings.PROJECT_DIR, 'uploaded_videos', saved_video_file)
            request.session['sequence_length'] = sequence_length
            return redirect('app:predict')
        else:
            return render(request, index_template_name, {"form": video_upload_form})

def predict_page(request):
    if request.method == "GET":
        if 'file_name' not in request.session:
            return redirect("app:home")
        if 'file_name' in request.session:
            video_file = request.session['file_name']
        if 'sequence_length' in request.session:
            sequence_length = request.session['sequence_length']
        path_to_videos = [video_file]
        video_file_name = video_file.split('\\')[-1]
        
        video_file_name_only = video_file_name.split('.')[0]
        print("Video: "+ video_file_name_only)
        start_time = time.time()
        print("<=== | Started Videos Splitting | ===>")
        preprocessed_images = []
        faces_cropped_images = []
        print(video_file)
        pred_inst.loadVideo(video_file)
        frames_images = pred_inst.getFrames()
        faces_images = pred_inst.getFaces()
        i = 1
        path = 'static/uploaded_images/'
        video_name = video_file_name_only.split('/')
        path += video_name[-1]
        size = None
        for frame in frames_images:
            height, width, layers = frame.shape
            size = (width,height)
            image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img = pImage.fromarray(image, 'RGB')
            image_path = path+"_preprocessed_"+str(i)+'.png'
            img.save(image_path)

            image_path = image_path.split('/')
            image_path = '/'.join(image_path[1:])
            preprocessed_images.append(image_path)
            i += 1

        pred = pred_inst.predictFaces()
        start_point = (1, 1)  
        end_point = (127, 127)
        thickness = 3
        for i,face in enumerate(faces_images):
            if pred[i] == 1:
                #make the rectangle with set tag of real
                faces_image = cv.cvtColor(face, cv.COLOR_BGR2RGB)
                image = cv.rectangle(copy.deepcopy(faces_image), start_point, end_point, (0, 255, 0), thickness)
                cv.putText(image, 'Real', (4, 21), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                #make the rectangle with set tag of fake
                faces_image = cv.cvtColor(face, cv.COLOR_BGR2RGB)
                image = cv.rectangle(copy.deepcopy(faces_image), start_point, end_point, (255, 0, 0), thickness)
                cv.putText(image, 'Fake', (4, 21), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            img = pImage.fromarray(image, 'RGB')
            image_path = path+"_cropped_faces_"+str(i)+'.png'
            img.save(image_path)
            image_path = image_path.split('/')
            image_path = '/'.join(image_path[1:])
            faces_cropped_images.append(image_path)

        if pred_inst.checkStatus():
            output = "REAL"
        else:
            output = "FAKE"
        print('Video Fle        : '+ video_file_name)
        video_file_name = video_file_name.split('/')
        return render(request, predict_template_name, {'preprocessed_images': preprocessed_images,"faces_cropped_images": faces_cropped_images, "original_video": video_file_name[-1], "output": output})
def about(request):
    return render(request, about_template_name)