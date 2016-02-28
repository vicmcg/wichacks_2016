import cv2
import glob
import numpy
import scipy.misc 
import os
#import antigravity


"""
WiCHACKS 2016
Victoria Scholl
Victoria McGowen
Elizabeth Bondi 
02/28/16
"""

def create_pokemon_eigenface_model(filenameList, threshold=100.0): 
   imageList = []
   labelList = numpy.arange(len(filenameList))

   for i, imName in enumerate(filenameList): 
      tmp = cv2.imread(imName, cv2.IMREAD_UNCHANGED)
      tmp[numpy.where(tmp[:,:,3]==0)] = 255
      tmpGrey = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
      imageList.append(tmpGrey)

   model = cv2.createEigenFaceRecognizer(threshold)
   model.train(imageList, labelList)

   return imageList, model


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
   threshold = 500
   filenameList = glob.glob(imageFolder+'/*.png')
   imageList, model= create_pokemon_eigenface_model(filenameList, threshold)
   
   newImageFilename = 'test_pokemon/mystery_pokemon.png'
   #newImageFilename = 'test_pokemon/pikachu_big.jpg'
   #newImageFilename = 'test_pokemon/eevee.png'
   #newImageFilename = 'test_pokemon/pokemon_2.png'
   #newImageFilename = 'test_pokemon/077.png'
   #newImageFilename = 'test_pokemon/niodoran_big.jpg'
   newImageFilename = 'test_pokemon/eevee2.png'

   newImage = cv2.imread(newImageFilename,cv2.IMREAD_UNCHANGED)
   print newImage.shape
   if newImage.shape[2] == 4: # if alpha channel
      tmp = cv2.imread(newImageFilename, cv2.IMREAD_UNCHANGED)
      tmp[numpy.where(tmp[:,:,3]==0)] = 255
      newImage = tmp
   newImage = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)

   # make image into a square if it's not
   if newImage.shape[0] != newImage.shape[1]: 
      newImage = square(newImage)

   newImageResized = scipy.misc.imresize(newImage,imageList[1].shape,interp='nearest')

   x,y = model.predict(newImageResized)
   print 'Index of closest pokemon: ', x
   nameFilename = 'name.txt'
   namesLUT = numpy.loadtxt(nameFilename, dtype='str')
   print namesLUT[x]

   # write out filename and image of matching pokemon 
   f = open('updates.txt', 'w')
   f.write(filenameList[x]+'\n')
   f.write(namesLUT[x])
   f.close()

   # display input image anc closest match
   cv2.imshow('Input Image', newImageResized)
   cv2.imshow('Closest Match: '+ namesLUT[x],cv2.imread(filenameList[x],0))
   cv2.waitKey(0)



