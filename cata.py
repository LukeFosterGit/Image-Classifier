from PIL import Image
import numpy
import os

from keras.models import load_model

# Loads in trained model to be used to catagorize.

model = load_model("C:\\Users\\Foste\\Desktop\\CV_Final\\final_model.h5") 

# Assigns names to the classes to the sub directories the model was trained on.

classes = {
    0: 'car',
    1: 'face',
    2: 'leaf',
    3: 'motorcycle',
    4: 'airplane'
}

# Last 100 images from each class to test the efficency of the model.

directory = "C:\\Users\\Foste\\Desktop\\CV_Final\\last100"

# Writes results to a txt file.

txt = open(r"C:\Users\Foste\Desktop\CV_Final\output.txt", "w") 

# Method that runs all of the images in the test directory against the saved h5 model and assigns them a class.

def classify(directory):   
    for subdir, dir, files in os.walk(directory):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".jpg"):
                    try:
                        image = Image.open(filepath)
                        image = image.resize((180,180))
                        image = numpy.expand_dims(image, axis = 0)
                        image = numpy.array(image)
                        pred = numpy.argmax(model.predict(image)[0], axis=-1)
                        sign = classes[pred]
                        f = os.path.join(file, sign) 
                        txt.write(file + " " + sign + "\n")
                        print(f) 
                    except:
                        # If image dimensions are inccorect the invalid image tag is assigned.
                        inv = "Invalid Image"
                        txt.write(file + "  " + inv + "\n")
                        print(file + "  " + inv + "\n")

# Call to method to classify the images in the target directory, and then save the changes to the output txt file.

classify(directory)
txt.close()