import matplotlib.pyplot as plt

class verticle:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return f"{self.name} ({self.x}, {self.y})"

    def plot(self):
        plt.annotate(self.name, (self.x, self.y), 
                     size=20, bbox=dict(boxstyle="circle,pad=0.2"),
                     horizontalalignment='center', verticalalignment='center',)

if __name__ == "__main__":
    p1 = verticle("A", 3, 4)
    print(p1)