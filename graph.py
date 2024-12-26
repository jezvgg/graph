from functools import singledispatch
import enum

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from verticle import verticle
from graph_types import graph_type


class Graph:
    '''
    Graph by matrix
    '''

    __verticles = []
    __edges = []
    matrix = []
    graph_type: enum.EnumMeta


    def __init__(self, verticles: dict, matrix: np.ndarray):
        self.graph_type = graph_type.SIMPLE
        if not np.all(matrix == matrix.T): self.graph_type |= graph_type.DIRECTED
        if matrix.max() > 1: self.graph_type |= graph_type.WEIGHTED
        self.verticles = verticles
        self.matrix = matrix


    @singledispatch
    def build(structs) -> "Graph":
        raise Exception("Конструктор не работает")
    

    @build.register
    @staticmethod
    def build_verticles(verticles: dict):
        matrix = []
        for vert in verticles:
            to = verticles[vert]
            matrix.append( [1 if x in to else 0 for x in verticles] )
        matrix = np.array(matrix)
        return Graph(verticles, matrix)


    @build.register
    @staticmethod
    def build_matrix(matrix: np.ndarray):
        return Graph([str(i) for i in range(matrix.shape[0])], matrix)


    @build.register
    @staticmethod
    def build_matrix(matrix: list):
        return Graph([str(i) for i in range(len(matrix))], np.array(matrix))


    def __eq__(self, __o: object) -> bool:
        return np.array_equal(self.matrix, __o.matrix)


    def __and__(self, __o: "Graph"):
        result = self.copy()
        result.matrix = (self.matrix+__o.matrix==2).astype(int)
        return result


    def __or__(self, __o: "Graph"):
        result = self.copy()
        result.matrix = (self.matrix+__o.matrix>0).astype(int)
        return result


    def __sub__(self, __o: "Graph"):
        result = self.copy()
        result.matrix = (self.matrix-__o.matrix>0).astype(int)
        return result


    def __xor__(self, __o: "Graph"):
        result = self.copy()
        result.matrix = (self.matrix!=__o.matrix).astype(int)
        return result


    def __str__(self):
        result = "  "+"".join([str(node) for node in self.verticles])+"\n"
        for i,row in enumerate(self.matrix):
            result += str(self.verticles[i]) + " " + "".join(map(str,row))+"\n"
        return result


    def __invert__(self):
        result = self.copy()
        result.matrix = (self.matrix==0).astype(int)
        return result


    def set(self, indexes: list[int]):
        '''
        Меняет вершины по заданым индексам.
        '''
        if len(indexes) != len(self.verticles):
            raise IndexError(f"indexes must be len of {len(self.verticles)}")

        new_matrix = self.matrix[indexes]
        self.matrix = new_matrix[:, indexes]
        self.__verticles = [self.verticles[x] for x in indexes]


    def replace(self, fro:str|verticle, to:str|verticle):
        '''
        Меняет 2 переданные вершины графа местами.
        '''
        index_fro = self.verticles.index(str(fro))
        index_to = self.verticles.index(str(to))
        indexes = [index_to if i==index_fro else index_fro if i==index_to else i for i in range(len(self.verticles))]
        self.set(indexes)


    def copy(self):
        return Graph(self.dict)


    def plot(self, x=0, y=0):
        '''
        x,y - смещение построения графа, если координаты вершин не заданы
        '''
        fig, ax = plt.subplots()

        if not isinstance(self.verticles[0], verticle):
            print("Перепись " + "="*32)
            space = np.linspace(np.pi/2, 2*np.pi+np.pi/2, len(self.verticles)+1)[:-1]
            self.__verticles = [verticle(name, np.cos(sp)+x, np.sin(sp)+y) for name, sp in zip(self.verticles, space)]

        print([vert.name for vert in self.verticles])
        print([vert.x for vert in self.verticles])
        print([vert.y for vert in self.verticles])

        for i,row in enumerate(self.matrix):
            for j,elem in enumerate(row):
                if elem == 0: continue
                if graph_type.DIRECTED not in self.graph_type and (j < i): continue

                if graph_type.DIRECTED in self.graph_type:
                    arrow = FancyArrowPatch((self.verticles[i].x, self.verticles[i].y),
                                                (self.verticles[j].x, self.verticles[j].y),
                                                connectionstyle="arc3,rad=.05", arrowstyle='->', mutation_scale=30)
                else:
                    x = (self.verticles[i].x - self.verticles[j].x) / 15
                    y = (self.verticles[i].y - self.verticles[j].y) / 15
                    arrow = FancyArrowPatch((self.verticles[i].x-x, self.verticles[i].y-y),
                                            (self.verticles[j].x+x, self.verticles[j].y+y),
                                            arrowstyle='<|-|>', mutation_scale=20)
                ax.add_patch(arrow)
                    # plt.plot([self.verticles[i].x, self.verticles[j].x], [self.verticles[i].y, self.verticles[j].y], c='black')

                if graph_type.WEIGHTED in self.graph_type:
                    va = 'bottom' if j >= i else 'top'
                    aligment = 'left' if j >= i else 'right'
                    ax.annotate(str(elem), (.5, .5), xycoords=arrow, ha='center', 
                                va=va, horizontalalignment=aligment, size=20)

        x = []
        y = []
        for v in self.verticles:
            x.append(v.x)
            y.append(v.y)
            v.plot()
        plt.scatter(x,y, c='red', zorder=1)



    @property
    def dict(self):
        review = {}
        if isinstance(self.__verticles[0],verticle): n_vert = [verticle(vert.name, vert.x, vert.y) for vert in self.verticles]
        else: n_vert = self.__verticles
        for vert, row in zip(n_vert, self.matrix):
            if isinstance(vert,verticle):
                review[vert] = [n_vert[i] for i,sym in enumerate(row) if sym]
            else:
                review[vert] = ''.join([self.__verticles[i] for i,sym in enumerate(row) if sym])
        return review


    @property
    def verticles(self):
        return self.__verticles


    @verticles.setter
    def verticles(self, verts):
        self.__verticles = verts


if __name__ == "__main__":
    # g = Graph({'a':'ad','b':'ad','c':'bc','d':'bc', 'e':'ac', 'f':'bd'})
    # h = Graph({'a':'bd','b':'bc','c':'bc','d':'ac'})
    # print("graph G1:",g,sep="\n")
    # print("graph H1:",h,sep="\n")
    # print("graph G1 and H1:",g&h,sep="\n")
    # print("graph G1 or H1:",g|h,sep="\n")
    # print("graph G1 / H1:",g-h,sep="\n")
    # print("graph G1 xor H1:",g^h,sep="\n")
    # print("Invert G1:",~g)
    # g.plot(1, 1)
    # h.plot(3.2, 1)
    g = Graph.build([[0, 10, 30, 50, 10],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 10],
                     [0, 40, 20, 0, 0],
                     [10, 0, 10, 30, 0]])
    print(repr(g.graph_type))
    g.plot()
    print(g)
    plt.show()