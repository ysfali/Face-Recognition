# Import the required modules
import cv2, os , sys
import numpy as np
from PIL import Image

# For face detection we will use the Haar Cascade provided by OpenCV.
# cascadePath = "haarcascade_frontalface_alt2.xml"
cascadePath = "/home/ysf/opencv/data/lbpcascades/lbpcascade_frontalface.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# For face recognition we will use the the LBPH Face Recognizer 
recognizer = cv2.face.createLBPHFaceRecognizer()
# recognizer = cv2.face.createEigenFaceRecognizer()

def read_images_and_labels(path) :
	images = []
	labels = []
	training = 0
	total_faces = 0
	ioerror = 0
	for dirname , dirnames , filenames in os.walk(path):
		for subdirname in dirnames :
			subject_path = os.path.join(dirname , subdirname)
			label = 0
			for letter in subdirname:
				label = label*10 + ord(letter)
			for filename in os.listdir(subject_path):
				try :
					training+=1
					# Read the image and convert to grayscale
					im = Image.open(os.path.join(subject_path,filename)).convert('L')
					# Convert the image format into numpy array
					image = np.asarray(im , dtype=np.uint8)
					# print 'filename : ',filename

					faces = faceCascade.detectMultiScale(image,minNeighbors=3)
					# # If face is detected, append the face to images and the label to labels
					for (x, y, w, h) in faces:
						total_faces+=1
						# Applying Histogram Equalization
						img = cv2.equalizeHist(image[y:y+h , x:x+w])
						images.append(img)
						labels.append(label)
						cv2.imshow("Adding faces to traning set...", img)
						# cv2.waitKey(100)


				except IOError :
					# print 'Name : ',subdirname,' Image :',filename
					# print "I/O error"
					# num_images-=1
					ioerror+=1
				except :
					print "Unexpected error : " , sys.exc_info()[0]
					raise
			# print 'Name : ',subdirname,' Images : ',num_images

	# return the images list and labels list
	print 'Training Images : ',training-ioerror
	print 'Total Faces : ',total_faces
	return images, labels

def read_test_images_and_labels(path) :
	c = 1
	images = []
	labels = []
	testing = 0
	ioerror = 0
	for dirname , dirnames , filenames in os.walk(path):
		for subdirname in dirnames :
			subject_path = os.path.join(dirname , subdirname)
			label = 0
			num_images = 0
			for letter in subdirname:
				label = label*10 + ord(letter)
			
			for filename in os.listdir(subject_path):
				try :
					testing+=1
					num_images+=1
					# Read the image and convert to grayscale
					im = Image.open(os.path.join(subject_path,filename)).convert('L')
					# Convert the image format into numpy array
					image = np.asarray(im , dtype=np.uint8)

					images.append(image)
					labels.append(label)

					# cv2.imshow("Reading test images...", image)
					# cv2.putText("Label :",c, (25,25), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
					# cv2.waitKey(50)


				except IOError :
					ioerror+=1
					num_images-=1
					# print "I/O error"
				except :
					print "Unexpected error : " , sys.exc_info()[0]
					raise
				
			# Get next Label
			c = c+1
	print 'Testing Images : ',testing-ioerror
	# return the images list and labels list
	return images, labels



print 'Reading Training Images'

# Path to Dataset
path = './faces96_train'
# Call the read_images_and_labels function and get the face images and the 
# corresponding labels
images , labels = read_images_and_labels(path)

labels = np.asarray(labels, dtype=np.int32)
cv2.destroyAllWindows()

print 'Making Recognizer Model'

# Perform the tranining
recognizer.train(images, labels)
#recognizer.train(np.asarray(images), np.asarray(labels))



print 'Reading Testing images'

test_path = './faces96_test'

test_images , test_labels = read_test_images_and_labels(test_path)
test_labels = np.asarray(test_labels,dtype=np.int32)


cv2.namedWindow('image')
# cap = cv2.VideoCapture(0)

print 'Predicting...'

count = 0

correct_prediction = 0
total_prediction = 0
wrong_prediction = 0

for predict_image in test_images:
	total_prediction+=1
	label = test_labels[count]
	count+=1
	faces = faceCascade.detectMultiScale(predict_image,minNeighbors=3)
	for (x, y, w, h) in faces:
		img = cv2.equalizeHist(predict_image)
		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
		cv2.imshow('image',img)
		result = cv2.face.MinDistancePredictCollector()
		recognizer.predict(img[y: y + h, x: x + w],result, 0)
		nbr_predicted = result.getLabel()
		conf = result.getDist()
		
		if(nbr_predicted == label):
			correct_prediction+=1
			# cv2.imshow("Recognizing Face", img[y: y + h, x: x + w])
			# cv2.putText(img,"Label : %f" %nbr_predicted , (25,25), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
			break
		else:
			wrong_prediction+=1
			# print 'Predicted : ',nbr_predicted,' Actual : ',label
	# cv2.waitKey(500)
			

print 'correct prediction : ',correct_prediction
print 'wrong prediction : ',wrong_prediction
print 'unpredicted : ',total_prediction-(correct_prediction+wrong_prediction)
print 'total prediction : ',total_prediction
print 'Percentage of correct prediction',100*correct_prediction//total_prediction

cv2.destroyAllWindows()
