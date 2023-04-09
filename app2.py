import tkinter as tk
import customtkinter as ctk
import keras
import tensorflow as tf
import numpy as np
import cv2 
from PIL import Image, ImageTk
import os
os.add_dll_directory(r"C:\Program Files\VideoLAN\VLC")
import vlc
import time

cap = cv2.VideoCapture(0)
model = keras.models.load_model(r"C:\Users\Chinh\DS Project\Untitled Folder\models\eff", compile=False)

def combine_funcs(*funcs):
    def combined_func(*args, **kwargs):
        for f in funcs:
            f(*args, **kwargs)
    return combined_func

#Prediction function, return the video frame with the predicted label and the bounding box as a numpy array
def predict(frame):
    #Flip the frame 180 degree
    frame = cv2.flip(frame, 1)
    frame = frame[50:500, 50:500,:]
    
    #Pre-proccessing
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(frame, (288,288))
    
    #Making prediction and set up our labels and bounding box coords
    pred = model.predict(np.expand_dims(resized, axis = 0), verbose = 0)
    if pred[0] < 0.5:
        label = 'awake'
    else:
        label = 'drowsy'
    coords = pred[1][0]
    
    #Add the label and bounding box to the existing frame
    if label == 'awake':

        #Display the bounding box
        cv2.rectangle(frame, 
                        tuple(np.multiply(coords[:2], [450, 450]).astype(int)),
                        tuple(np.multiply(coords[2:], [450, 450]).astype(int)), 
                        (0,0,255), 2)
        
        #Display the background for the label
        cv2.rectangle(frame, 
                          tuple(np.add(np.multiply(coords[:2], [450,450]).astype(int), 
                                        [0,-30])),
                          tuple(np.add(np.multiply(coords[:2], [450,450]).astype(int),
                                        [100,0])), 
                                (0,0,255), -1)
        
        #Display the label
        cv2.putText(frame, label, tuple(np.add(np.multiply(coords[:2], [450,450]).astype(int),
                                                       [0,-5])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    else:
        #Display the bounding box 
        cv2.rectangle(frame, 
                        tuple(np.multiply(coords[:2], [450, 450]).astype(int)),
                        tuple(np.multiply(coords[2:], [450, 450]).astype(int)), 
                        (255,0,0), 2)
        
        #Display the background for the label
        cv2.rectangle(frame, 
                          tuple(np.add(np.multiply(coords[:2], [450,450]).astype(int), 
                                        [0,-30])),
                          tuple(np.add(np.multiply(coords[:2], [450,450]).astype(int),
                                        [110,0])), 
                                (255,0,0), -1)
        
        #Display the label
        cv2.putText(frame, label, tuple(np.add(np.multiply(coords[:2], [450,450]).astype(int),
                                                       [0,-5])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
    #Return the frame with the prediction labels and bounding box
    return frame, pred[0][0]

app = tk.Tk()
app.geometry('600x600')
app.title('Drowsy Detector')
ctk.set_appearance_mode('dark')
score = 0

def reset_score():
    global score
    score = 0

def alert():
    alert = vlc.MediaPlayer("C:/Users/Chinh/DS Project/Untitled Folder/1.mp3")
    alert.play()

def tab1():
    global score
    def tab2():
        threshold = define_threshold()
        option_bar.destroy()
        tab1_button.destroy()

        vidFrame = tk.Frame(height=450, width=450)
        vidFrame.pack()
        vid = ctk.CTkLabel(vidFrame)
        vid.pack()

        scoreLabel = ctk.CTkLabel(text = score, height = 40, width = 120, font = ('Arial', 20), text_color='white', fg_color='black', master=app)
        scoreLabel.pack(pady = 10)

        #Create a reset button for score
        resetButton = ctk.CTkButton(text = 'RESET', command = reset_score, height = 40, width = 120, font = ('Arial', 20), text_color='white', fg_color='red', master=app)
        resetButton.pack()

        def detect():
            global score
            #Get frame
            _, frame = cap.read()

            #Make prediction from the frame
            predict_frame, prob = predict(frame)

            #Convert the prediction frame from an numpy array format into ImageTk format
            imgarr = Image.fromarray(predict_frame)
            imgtk = ImageTk.PhotoImage(imgarr)

            #Display the image onto the vid frame
            vid.imgtk = imgtk
            vid.configure(image = imgtk)
            vid.after(10, detect)

            #Scoring system
            if prob > 0.5:
                score = score + prob
            else:
                if score < threshold:
                    reset_score()
            
            if score > threshold:
                alert()
                reset_score()

        
            scoreLabel.configure(text = score)

        detect()

    state_options = [
    "Awake",
    "A bit sleepy",
    "Sleepy",
    "Very sleepy"
]

    state_var = tk.StringVar(app)
    state_var.set(state_options[0])

    #Create an option bar for state input
    option_bar = tk.OptionMenu(app, state_var, *state_options)
    option_bar.pack()

    def define_threshold():
        current_state = state_var.get()

        if current_state == 'Awake':
            threshold = 30

        elif current_state == 'A bit sleepy':
            threshold = 15

        elif current_state == 'Sleepy':
            threshold = 5

        elif current_state == 'Very sleepy':
            threshold = 2
        return threshold
    threshold = define_threshold()

    tab1_button = tk.Button(app, text='Submit', command = tab2)
    tab1_button.pack()

tab1()

app.mainloop()
