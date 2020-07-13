
### IMPORTS ###

import numpy as np
import math
import cv2

### DEFINITIONS ###
k = 16
thres = 3
block_size = 16
filename = "video_sample.mp4"

### FUNCTIONS ###

# Function that fixes the sizes of a frame to be multiples of block_size

def fixFrameSize(frame, block_size):
    
    # Define Height and Width so that they are both multiples of block_size
    H_extra = (block_size - frame.shape[0] % block_size)  % block_size
    W_extra = (block_size - frame.shape[1] % block_size) % block_size
    
    # If needed, the extra lines/columns will be padded with zeros (filled with black)
    frame = np.pad(frame, ((0,H_extra),(0,W_extra)), 'constant')
    
    return frame

# Function that calculates motion vectors for all the blocks of a given frame based on a source frame

def calcMotionVectors(srcFrame, currFrame, k, block_size):
      
    vectors = []
    
    H = srcFrame.shape[0]
    W = srcFrame.shape[1]

    # Iterate through every block in the frame
    for i in range(0, H, block_size):
        for j in range(0, W, block_size):
            
            currBlock = currFrame[i:i+block_size, j:j+block_size]
            
            # Keep lists with SAD metrics and corresponding indices of blocks
            sad_metrics = []
            idxs = []
            
            # Iterate through every adjacent (based on k) block
            for m in range(i-k, i+block_size+k):
                for n in range(j-k, j+block_size+k):
                    
                    if(m < 0 or n < 0 or m > H - block_size or n > W - block_size):
                        continue
                    
                    checkingBlock = srcFrame[m:m+block_size, n:n+block_size]
                    
                    # Calculate the SAD metric and store the indices
                    sad = np.sum(np.abs(np.subtract(currBlock,checkingBlock)))
                    sad_metrics.append(sad)
                    idxs.append((n,m))
            
            # Find the best-matching block from the source frame
            minSAD = min(sad_metrics)
            minIdx = sad_metrics.index(minSAD)
            
            # Based on its indices, create the motion vector as such: [Xo, Yo, X, Y]
            matchingBlockX, matchingBlockY = idxs[minIdx]
            blockVector = [matchingBlockX, matchingBlockY, j,i]
            
            vectors.append(blockVector)
    
    return vectors

# Function that replaces a block from a frame with the corresponding block of the background, essentially hiding moving objects

def hideObject(frame, background, blockIdx, block_size):
    
    W = frame.shape[1]
    
    i = int ((blockIdx // (W / block_size))*block_size)
    j = int ((blockIdx % (W / block_size))*block_size)
    
    frame[i:i+block_size, j:j+block_size] = background[i:i+block_size, j:j+block_size]
    
    return frame


### DRIVER CODE FOR QUESTION B ### 

# Open the video file
cap = cv2.VideoCapture(filename)

# Store all the video frames in this list
all_frames = []

# Render all the frames and store them (in grayscale)
while(True):
    ret, frame = cap.read()
    if ret == True:
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        all_frames.append(frame)
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        
        break

# Fix the sizes
for i in range(len(all_frames)):
    all_frames[i] = fixFrameSize(all_frames[i], block_size)

# Define the first frame of the video as the background
background = all_frames[0]

vectors = []

# Calculate the motion vectors for all frames and hide objects from them
for i in range(1, len(all_frames)):
    
    print("Calculating Motion Vectors for Frame ", i)
    vectors.append(calcMotionVectors(all_frames[i-1], all_frames[i], k, block_size))
    
# After the calculation, hide objects where needed
for i in range(1, len(all_frames)):
    
    for j in range(len(vectors[i-1])):
        
        Xo = vectors[i-1][j][0]
        Yo = vectors[i-1][j][1]
        X = vectors[i-1][j][2]
        Y = vectors[i-1][j][3]
        
        # If the block has moved
        if ( (abs(X - Xo) >= thres) or (abs(Y - Yo) >= thres) ):
            
            # Replace it with the corresponding background block
            all_frames[i] = hideObject(all_frames[i], background, j, block_size)

# Display the results
for f in all_frames:
    cv2.imshow('video',f)
    cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()