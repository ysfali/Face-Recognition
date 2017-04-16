import os

#path to training or testing set
path = "faces96_test"
# path = "faces96_train"

for dirname , dirnames , filenames in os.walk(path):
	for subdirname in dirnames :
		subject_path = os.path.join(dirname , subdirname)
		for filename in os.listdir(subject_path):
			name = filename.split('.')
			if len(name)==3 and name[0]!='' :
				print filename
				num = name[1:2]
				n = int(num[0])
				case = path.split('_')
				if case[1]=='test'
					if n>5 :
						os.remove(os.path.join(subject_path,filename))
				elif case[1]=='train'
					if n<6 :
						os.remove(os.path.join(subject_path,filename))
				# print name;