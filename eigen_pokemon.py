import cv2
import glob
import numpy


# specify directory for 
imageFolder = 'pokemon'
filenameList = glob.glob(imageFolder+'/*.png')

# create eigenface model 
model = cv2.createEigenFaceRecognizer(threshold=100.0)
imageList = []
labelList = numpy.arange(len(filenameList))


for i, imName in enumerate(filenameList): 
   imageList.append(cv2.imread(imName,0))
   
   #cv2.imshow('current image',cv2.imread(imName))
   #cv2.waitKey(0)

model.train(imageList, labelList)

print imageList
print labelList

# create funtion that takes in unknown image, searches through model
# for matching known pokemon, determines name from integer, returns name
newImage = cv2.imread('mystery_pokemon.png',0)

# determine integer of unknown pokemon 
x,y = model.predict(newImage)
print x

# make list of pokemon names & corresponding integers