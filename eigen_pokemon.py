import cv2
import glob
import numpy
import scipy.misc 
#import antigravity

def create_pokemon_eigenface_model(imageFolder='pokemon', threshold=100.0): 
   filenameList = glob.glob(imageFolder+'/*.png')
   imageList = []

   for i, imName in enumerate(filenameList): 
      imageList.append(cv2.imread(imName,0))
   
   model = cv2.createEigenFaceRecognizer()#threshold)
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

def square(im): 


   if im.shape[0] > im.shape[1]:
      r1 = (im.shape[0]-im.shape[1])/2 
      if (im.shape[0]-im.shape[1])%2 != 0: 
         r2 = r1 + 1
      else: 
         r2 = r1
      newImage = numpy.pad(im, ([0,0],[r1,r2]), 'constant', 
         constant_values=(255,255))

   else: 
      r1 = (im.shape[1]-im.shape[0])/2 
      if (im.shape[1]-im.shape[0])%2 != 0: 
         r2 = r1 + 1
      else: 
         r2 = r1
      newImage = numpy.pad(im, ([r1,r2],[0,0]), 'constant', 
         constant_values=(255,255))  

   return newImage   

if __name__ == '__main__':

   imageFolder = 'pokemon'
   threshold = 100
   imageList, model= create_pokemon_eigenface_model(imageFolder, threshold)
   
   newImageFilename = 'test_pokemon/mystery_pokemon.png'
   newImageFilename = 'test_pokemon/pikachu_big.jpg'
   newImageFilename = 'test_pokemon/eevee.png'
   newImageFilename = 'test_pokemon/pokemon_2.png'
   newImageFilename = 'test_pokemon/077.png'

   newImage = cv2.imread(newImageFilename,0)
   #newImageResized = resize(imageList, newImage)

   if newImage.shape[0] != newImage.shape[1]: 
      newImage = square(newImage)

   newImageResized = scipy.misc.imresize(newImage,imageList[1].shape,interp='nearest')

   x,y = model.predict(newImageResized)
   print 'Index of closest pokemon: '
   print x

   nameFilename = 'name.txt'
   namesLUT = numpy.loadtxt(nameFilename, dtype='str')
   print namesLUT[x]

   # write out filename and image of matching pokemon 
   f = open(updates.txt, 'w')
   


