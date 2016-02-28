import cv2
import glob
import numpy


def create_pokemon_eigenface_model(imageFolder='pokemon', threshold=100.0): 
   filenameList = glob.glob(imageFolder+'/*.png')
   imageList = []
   labelList = numpy.arange(len(filenameList))

   for i, imName in enumerate(filenameList): 
      imageList.append(cv2.imread(imName,0))
   
   model = cv2.createEigenFaceRecognizer(threshold=50.0)
   model.train(imageList, labelList)

   return imageList, model

def resize(imageList, newImage):

   print 'Database pokemon image dimensions: '
   imRows = imageList[1].shape[0]
   print 'rows: ', imRows
   imColumns = imageList[1].shape[1]
   print 'columns: ', imColumns

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
      r = (newColumns - imColumns)/2
      newImage = newImage[:,r:r+imRows]

   # take care of mismatched image dimensions by padding/cropping
   print 'AFTER PADDING: '
   newRows = newImage.shape[0]
   print 'rows: ', newRows
   newColumns = newImage.shape[1]
   print 'columns: ', newColumns
   cv2.imshow('after padding', newImage)
   cv2.waitKey(0)

   return newImage


if __name__ == '__main__':

   imageFolder = 'pokemon'
   threshold = 10
   imageList, model = create_pokemon_eigenface_model(imageFolder, threshold)
   
   newImageFilename = 'mystery_pokemon.png'
   #newImageFilename = 'pikachu_big.jpg'
   newImage = cv2.imread(newImageFilename,0)
   newImageResized = resize(imageList, newImage)

   x,y = model.predict(newImageResized)
   print x

   nameFilename = 'name.txt'
   namesLUT = open(nameFilename).read()
   print namesLUT


