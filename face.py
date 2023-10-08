import cv2

class Face:
    def __init__(self, path, age,size=100):
        self.path = path
        self.age = age
        self.size = size
    def mod(self):
        self.image = cv2.imread(self.path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.resize(self.image, (self.size, self.size))
        return self.image
    
    def show(self):
        cv2.imshow('gray', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


