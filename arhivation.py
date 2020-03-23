class psi_curve:
    def __init__(self, size=60, n=2, gamma=10, limit=15):
        """size is a linear size of required lattice, mean, that if size==10, that lattice is done as
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] ^ 2
        
        so, if your date is in limits of [5, 10], you need to shift it to [0, 5] and use size = 6"""
        self.n = n
        self.gamma = gamma
        self.limit = limit
        self.a = (1 / gamma) / (gamma - 1)

        x = np.arange(size)
        #endpoint=True breaks the plot of curve, 
        #but allows work with [0, 1], not [0, 1)  
        self.coord = []
        for i in x:
            for j in x:
                result = psi.inner_function((i / size, j / size), gamma, n)[0]        
                self.coord.append((i, j, result))
        self.coord.sort(key=lambda x: x[2])
        self.x = np.array([i[0] for i in self.coord])
        self.y = np.array([i[1] for i in self.coord])
        
    def distance_from_coordinates(self, x):
        return np.where(np.logical_and((self.x == x[0]), (self.y == x[1])))[0][0]
    
    
def toLatticeFunction(x, y):
    return (np.int(np.round((lin_size - 1) * x)), np.int(np.round((lin_size - 1) * y)))

def sub_archiver(full_size, regress_size, arrays):
    result = []
    for array in arrays:
        sqs = []
        slopes = []
        ints = []
        x = np.fromiter(range(regress_size), np.int32, count=-1)
        for y in np.split(np.array(array), full_size // regress_size):
            sq, slope, intercept = np.polyfit(x, y, 2)
            #slope, intercept = np.polyfit(x, y, 1)

            sqs.append(sq)
            slopes.append(slope)
            ints.append(intercept)

        result.append(sqs)
        result.append(slopes)
        result.append(ints)
    return result
    
def sub_dearchiver(full_size, regress_size, arrays):
    result = []
    for sqs, slopes, ints in zip(arrays[::3], arrays[1::3], arrays[2::3]):
#    for slopes, ints in zip(arrays[::2], arrays[1::2]):
        archx = list(range(full_size))
        archy = [ints[i // regress_size] + slopes[i // regress_size] * (i % regress_size)
                 + sqs[i // regress_size] * (i % regress_size) * (i % regress_size) for i in range(full_size)]
        #archy = [ints[i // regress_size] + slopes[i // regress_size] * (i % regress_size) for i in range(full_size)]

        result.append(archy)
    return result

class Approximator():
    def __init__(self, lin_size):
        self.array = np.zeros((lin_size, lin_size)) #powerfull machine learning technic
        self.curve = psi_curve(size=lin_size)
        self.lin_size = lin_size
        
    def predict(self, X):
        y = []
        for x in X:
            y.append(self.array[toLatticeFunction(x[0], x[1])])
        return y

    def fit(self, X, Y):
        subarray = np.zeros_like(self.array)
        for x, y in zip(X, Y):
            self.array[toLatticeFunction(x[0], x[1])] += y
            subarray[toLatticeFunction(x[0], x[1])] += 1
        for i in range(lin_size):
            for j in range(lin_size):
                self.array[i, j] /= max(1, subarray[i, j])
    
    def archiver(self):
        h = []
        for i in range(app.lin_size):
            for j in range(app.lin_size):
                h.append((app.curve.distance_from_coordinates((i, j)), app.array[i, j]))
        self.h = np.array(sorted(h, key=lambda x: x[0]))[:, 1]
        #level 1
        self.level1 = sub_archiver(lin_size * lin_size, lin_size // 10, [self.h])
        #level 2
        self.level2 = sub_archiver(lin_size * 10, lin_size // 10, self.level1)
        #level 3
        self.level3 = sub_archiver(10 * 10, 10, self.level2)
        #level 4
        self.level4 = sub_archiver(10, 10, self.level3)

    def dearchiver(self):
        #level 4
        self.level3 = sub_dearchiver(10, 10, self.level4)
        #level 3
        self.level2 = sub_dearchiver(10 * 10, 10, self.level3)
        #level 2
        self.level1 = sub_dearchiver(lin_size * 10, lin_size // 10, self.level2)
        #level 1
        self.h = sub_dearchiver(lin_size * lin_size, lin_size // 10, self.level1)[0]
        #from h to array
        self.array = np.zeros((lin_size, lin_size))
        for x, y in zip(range(lin_size * lin_size), self.h):
            self.array[curve.coord[x][0], curve.coord[x][1]] = y