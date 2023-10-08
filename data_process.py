from face import Face
import numpy as np
import os
import time
dir = 'face_age'  

X = []
Y = []
start_time = time.time()
for age in range(1, 101):
    try:
        
        folder_name = f"{age:03d}"
        folder_path = os.path.join(dir, folder_name)
        print(f"Processing folder: {time.time() - start_time} seconds")
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            try:
                facs = Face(file_path, age,size=200)
                X.append(facs.mod())
                Y.append(facs.age)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    except Exception as e:
        print(f"Error processing {folder_path}: {e}")
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")
def save(X,Y):
    np.save('X.npy', X)
    np.save('Y.npy', Y)
X = np.array(X)
Y = np.array(Y)
# print(X.shape)
# print(Y.shape)
save(X,Y)