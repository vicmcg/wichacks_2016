import cv2
import glob
import numpy


#def create_pokemon_database(imageFolder): 
imageFolder = 'pokemon'
filenameList = glob.glob(imageFolder+'/*.png')
imageList = []
labelList = numpy.arange(len(filenameList))
for i, imName in enumerate(filenameList): 
   imageList.append(cv2.imread(imName,0))


# create eigenface model 
model = cv2.createEigenFaceRecognizer(threshold=50.0)
# train model based on database
model.train(imageList, labelList)

print 'Database pokemon image dimensions: '
imRows = imageList[1].shape[0]
print 'rows: ', imRows
imColumns = imageList[1].shape[1]
print 'columns: ', imColumns

# input mystery image
newImage = cv2.imread('mystery_pokemon.png',0)
newImage = cv2.imread('pokemon_2.png',0)
print 'Mystery pokemon image dimensions: '
newRows = newImage.shape[0]
print 'rows: ', newRows
newColumns = newImage.shape[1]
print 'columns: ', newColumns

if newRows < imRows:
   r1 = (imRows-newRows)/2
   if (imRows-newRows)%2 != 0: 
      r2 = r1 + 1
   else: 
      r2 = r1
   newImage = numpy.pad(newImage, ([r1,r2],[0,0]), 'constant', 
      constant_values=(255,255))

elif newRows > imRows:
   r = (newRows - imRows)/2
   newImage = newImage[r:r+imRows,:]

if newColumns < imColumns:
   r1 = (imColumns-newColumns)/2 
   if (imColumns-newColumns)%2 != 0: 
      r2 = r1 + 1
   else: 
      r2 = r1
   newImage = numpy.pad(newImage, ([0,0],[r1,r2]), 'constant', 
      constant_values=(255,255))

elif newColumns > imColumns: 
   r = (newRows - imRows)/2
   newImage = newImage[:,r:r+imRows]

# take care of mismatched image dimensions by padding/cropping
print 'AFTER PADDING: '
newRows = newImage.shape[0]
print 'rows: ', newRows
newColumns = newImage.shape[1]
print 'columns: ', newColumns
cv2.imshow('after padding', newImage)
cv2.waitKey(0)


# create funtion that takes in unknown image, searches through model
# for matching known pokemon, determines name from integer, returns name


# determine integer of unknown pokemon 
x,y = model.predict(newImage)

print x

# make list of pokemon names & corresponding integers
# read in RTF file