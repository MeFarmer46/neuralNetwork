import pickle

class objectSave():
    def __init__(self, path):
        self.path = path
    def write(self, input):
        with open(self.path, 'wb') as file:
            pickle.dump(input, file)
    def open(self):
        with open(self.path, 'rb') as file:
            data = pickle.load(file)
        return data