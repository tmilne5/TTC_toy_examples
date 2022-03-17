import imageio
import os
import sys

#take path with jpegs in it as input
path = sys.argv[1]
filenames = os.listdir(path)
filenames = [file for file in filenames if file[-3:] =='jpg']
filenames.sort(key = lambda name: int(name[4:-4]))

print(filenames)
#Run this code, saving it to the right path

images = []
for filename in filenames:
    images.append(imageio.imread(os.path.join(path,filename)))
imageio.mimsave(os.path.join(path,'movie.gif'), images)