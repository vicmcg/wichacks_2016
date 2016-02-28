import numpy
import cv2
import glob
import os
import scipy.misc
#import antigravity

def eigenfaces(faceImages,imRows,imColumns,newFace,knownFaceThreshold):
   """
   title::
      eigenfaces

   description:
         Implementation of facial recognition using eigenfaces from
      the paper by M. Turk and A. Pentland, 1991. 
         A series of M faces (each n-rows by m- columns) from a database 
      are read in and vectorized to be columns of a numpy N-D array of 
      size n*m rows by M columns, called 'faceImages.' 
         An average face is calculated from the database faces,
      and then the difference of each face from the average is calculated.
      The resulting difference vectors are stored in array, A.
         A matrix is then contructed from the difference array, A.T * A.
      (This is a shortcut based on linear algebra - instead of constructing
      a covariance matrix of size n*m x n*m, we instead use a matrix of
      size MxM to find M-1 meaningful eigenvalues and eigenvectors.)
         The eigenvalues and eigenvectors are computed using numpy.linalg.eig.
      The eigenvectors are then sorted into ascending order based on their
      corresponding eigenvalues. 
         Next, the new face is projected onto the eigenfaces to compute a set
      of weights (describing the contribution of each eigenface in representing
      the new face). Weights are also computed for each database face.
         The distance between the new face and every face in the database is
      calculated. If any of the distances are less than the knownFaceThreshold
      value, then the face exists in the database (it is recognized!)
         The 10 best matches to the input image are stored in an array for
      display. 
         The input face image is then reconstructed using a weighted sum of
      the eigenfaces. It is placed on the end of the array along with the best
      matches. This array is returned, called "matches." 
                  

   attributes::
      faceImages
         Vectorized database face images, each a column in this N-D
         numpy array. 

      imRows
         Number of rows in the input and database face images 
         (112 for the RIT and AT&T databases).

      imColumns
         Number of columnss in the input and database face images 
         (92 for the RIT and AT&T databases).

      newFace
         Input image of type N-D numpy array to be matched with and 
         reconstructed using faces from the specified database set.

      knownFaceThreshold
         Distance threshold that indicates the input face is known within
         the database. 

   returns::
      matches
         Numpy array containing the top 10 best matching database
         face images with the input image (displayed in a row
         ranging from 1st best to 10th best match from left to right) and
         the reconstruction using eigenfaces (11th image to the right).

   author::
      Victoria Scholl
      Image Processing & Computer Vision II
      Course Directed project 2
      04/26/15

   """

   # compute eigenfaces
   faceImages = 1.0 * faceImages
   numberFaces = faceImages.shape[0]
   avg = numpy.sum(faceImages, axis=0) / (1.0 * numberFaces)
   cv2.imwrite('average_face.tif',avg.reshape((imRows,imColumns)).astype('uint8'))
   differences = faceImages - avg
   A = numpy.asmatrix(differences).T  # images are column vectors
   eigenValues, eigenVectors = numpy.linalg.eig(A.T*A)
  
   # Sort eigenvectors based on ascending eigenvalues
   indices = numpy.argsort(eigenValues)
   indices = indices[::1]
   eigenValues = eigenValues[indices]
   print indices.shape

   # compute eigenfaces. 
   eigenfaces = A * eigenVectors
   print 'eigenVectors: ', eigenVectors.shape
   print 'eigenFaces: ', eigenfaces.shape
   # normalize eigenfaces by vector magnitudes
   normEigenfaces = eigenfaces / numpy.linalg.norm(eigenfaces,axis=0) 
   test =  eigenfaces[:,0]/numpy.sqrt(numpy.sum(numpy.square(eigenfaces[:,0])))
   print numpy.sqrt(numpy.sum(numpy.square(test)))
   
   # store 10 first eigenfaces to display
   tenEigenfaces = numpy.zeros((imRows,10*imColumns))
   for i in range(0,10):
      positive = normEigenfaces[:,i] + abs(numpy.min(normEigenfaces[:,i]))
      normScaled = (positive / numpy.max(positive)) * 255
      tenEigenfaces[0:imRows,i*imColumns:imColumns*(i+1)] = \
            normScaled.reshape((imRows,imColumns)).astype('uint8')
   
   cv2.imshow('10 first eigenfaces', tenEigenfaces.astype('uint8'))
   cv2.imwrite('first_10_eigenfaces_rit.tif', tenEigenfaces.astype('uint8'))
   cv2.waitKey(0)
   
   ## STEP 2: project new face into face space
   print 'new face dims: ', newFace.shape
   print 'avg dims: ', avg.shape
   print 'eigenfaces.T shape: ', (normEigenfaces.T).shape
   print 'new face dif dims:' , numpy.asmatrix(newFace - avg).T.shape
   newFaceDifferences = numpy.asmatrix(newFace - avg).T
   newFaceWeights = normEigenfaces.T * newFaceDifferences
   print 'new face weights dims: ', newFaceWeights.shape 
   databaseWeights = normEigenfaces.T * A

   # computes distance using weights to determine if the input face is known
   distances = numpy.sqrt(numpy.sum(numpy.square( \
                  databaseWeights - newFaceWeights),axis=0))
   sortedDistances = numpy.argsort(distances)
   distances2 = numpy.copy(distances).reshape((sortedDistances.shape[1]))
   print 'Distances, reshaped DIMS: ', distances2.shape
   distances = distances2[sortedDistances]
   faceImages = faceImages[sortedDistances]
      
   if distances[0,0] < knownFaceThreshold:
      print 'Face exists already in database!'
   else:
      print 'Face does not exist in database!'

   matches = numpy.zeros((imRows,11*imColumns))
   i = 0
   while i < 10: 
      print 'Epsilon: ',distances[0,i]
      #cv2.imshow('closest match ', \
      #faceImages[:,i].reshape((imRows,imColumns)).astype('uint8'))
      #cv2.waitKey(0)
      matches[0:imRows,imColumns*i:imColumns*(i+1)] = \
              faceImages[:,i].reshape((imRows,imColumns))
      i += 1
   
   # reconstruct new input face using weighted sum of eigenfaces
   newFaceProjection  = newFaceWeights.T * normEigenfaces.T
   reconstruction = newFaceProjection + avg
   reconstruction = numpy.clip(reconstruction.reshape((imRows,imColumns)),\
                               0,255).astype('uint8')
   # cv2.imshow('Reconstructed', numpy.clip(\
   #      reconstruction.reshape((imRows,imColumns)),0,255).astype('uint8'))
   
   # combine matches and reconstruction into array 
   matches[0:imRows,imColumns*i:imColumns*(i+1)] = reconstruction

   return matches

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

   faces = []
   imageFolder = 'pokemon'
   filenameList = glob.glob(imageFolder+'/*.png')
   for i, imName in enumerate(filenameList): 
      faces.append(cv2.imread(imName,0).flatten())

   currentImage = cv2.imread(imName,0)
   numberRows = currentImage.shape[0]
   numberColumns = currentImage.shape[1]
   faces = numpy.asarray(faces).reshape((len(filenameList),numberRows*numberColumns))

   # loop through series of input images and return matches & reconstructions
   filenames = ['test_pokemon/meowth.png'] #'test_pokemon/077.png', 
   knownFaceThreshold = 1.0
   for filename in filenames:
      print filename
      newFace = cv2.imread(filename,0)
      cv2.imshow('square',newFace)
      cv2.waitKey(0)

      if newFace.shape[0] != newFace.shape[1]: 
         newFace = square(newFace)
      cv2.imshow('square',newFace)
      cv2.waitKey(0)
      newFace = scipy.misc.imresize(newFace,(numberRows,numberColumns),interp='nearest')

      newFace = newFace.flatten()
      cv2.imshow('new face', \
            (newFace.reshape((numberRows,numberColumns))).astype('uint8'))
      matches = eigenfaces(faces,numberRows,numberColumns,newFace,
                                knownFaceThreshold)
      cv2.imshow('Matches and reconstruction for '+ filename, \
                  matches.astype('uint8'))
      cv2.waitKey(0)
      
   testKnownFace = True
   if testKnownFace == True:
      newFace = faces[43,:].flatten()
      filename = 'beard_guy'
      #newFace = faces[2,:].flatten()
      #filename = 'Carl.jpg'
      cv2.imshow('new face', (newFace.reshape((numberRows,numberColumns))).astype('uint8'))
      cv2.waitKey(0)
      knownFaceThreshold = 1.0
      matches = eigenfaces(faces,numberRows,numberColumns,newFace,knownFaceThreshold)
      cv2.imshow('Matches for ' + filename, matches.astype('uint8'))
      writeFilename = 'matches_'+database+'_' + filename[0:-4] + '.tif'
      print 'matches written to file: ', writeFilename
      cv2.imwrite(writeFilename, matches)
      cv2.waitKey(0)

