import enum

@enum.unique
class graph_type(enum.IntFlag):
    SIMPLE = 0
    WEIGHTED = 1
    DIRECTED = 2