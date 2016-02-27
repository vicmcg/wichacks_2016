import cv2

if __name__ == '__main__':
	filename = 'pokemon/001.png'
	im = cv2.imread( filename )
	cv2.imshow('hi', im)
	cv2.waitKey(0)
	numRow, numCol, numBand = im.shape
	print numRow, numCol, numBand
