import numpy as np

class Line:

    def __init__(self, coor1, coor2):
        self.coor1 = coor1
        self.coor2 = coor2

    def distance(self):
        dist_x = (self.coor1[0] - self.coor2[0])**2
        dist_y = (self.coor1[1] - self.coor2[1])**2
        euc = np.sqrt(dist_y + dist_x)
        return euc

    def slope(self):
        pass


coordinate1 = (3,2)
coordinate2 = (8,10)

li = Line(coordinate1,coordinate2)

print(li.distance())