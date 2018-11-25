import PIL as PIL
from PIL import Image, ImageTk, ImageDraw
import cv2
from tkinter import *
import numpy as np
from neuralnetwork.predictor import get_prediction
import datetime

# webcam settings - setting sizes is not working for all webcams
width, height = 1000, 540
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# base of the GUI
root = Tk()

# code to bind ESCAPE key for quit, SPACE for snapshot
k = 0


def keypress(key):
    global k
    print('key pressed')
    if key.char == ' ':
        k = 1


root.bind('<Escape>', lambda e: root.quit())
root.bind("<KeyPress>", keypress)

# create a label and pack it to the gui
lmain = Label(root)
lmain.pack()


# split and resize an image
def split_and_resize(img):
    img_resized_width = 200
    img_width = img.width
    img_height = img.height
    img_resized_height = round((img_height / img_width) * img_resized_width)

    img = img.resize((img_resized_width, img_resized_height), PIL.Image.ANTIALIAS)
    img1 = img.crop((0, 0, img_resized_width / 2, img_resized_height))
    img2 = img.crop((img_resized_width / 2, 0, img_resized_width, img_resized_height))
    return img, img1, img2


# remove black boxes added by OpenCV
def remove_black_boxes(img):
    return img.crop((25, 0, img.width, img.height))


# prepare an image to run through the predictor
def preprocess(img):
    img, img1, img2 = split_and_resize(img)
    img1 = remove_black_boxes(img1)
    img2 = img2.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    img2 = remove_black_boxes(img2)

    img1 = np.asarray(img1.getdata(), dtype=np.int).reshape((img1.size[1], img1.size[0]))
    img2 = np.asarray(img2.getdata(), dtype=np.int).reshape((img2.size[1], img2.size[0]))

    # normalize greyscale values to be [0-1]
    img1 = img1 / 255
    img2 = img2 / 255
    return img, img1, img2


# save current snapshot and run it through prediction process
def process(img):
    img = img.convert('L')
    # (640, 480)
    # (960, 540)
    img_std_aspect_ratio = 960 / 540  # aspect ration for training/testing data

    img, img_left, img_right = preprocess(img)

    values = np.array([img_left, img_right])
    values = values.reshape([values.shape[0], values.shape[1], values.shape[2], 1])
    print(values.shape)
    return get_prediction(values)


# show a new frame: take webcam snapshot and write it to the gui
def show_frame():
    global k
    # take snapshot and transform result to a color image
    _, frame = cap.read()

    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

    img = PIL.Image.fromarray(cv2image)
    img_width = img.width
    img_height = img.height
    draw = ImageDraw.Draw(img)
    draw.line((img_width / 2 - 1, 0, img_width / 2 + 1, img_height), fill=0)

    # transform image for usage in Tkinter and write it to the label
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)

    # create clean image
    img = PIL.Image.fromarray(cv2image)
    if k == 1:  # if SPACE pressed
        start = datetime.datetime.now()
        process(img)
        end = datetime.datetime.now()
        print('request took: ' + str(end - start) + ' to complete ')
        k = 0

    lmain.after(10, show_frame)  # execute this method again after 10 milliseconds


show_frame()
root.mainloop()
