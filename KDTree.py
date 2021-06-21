class KDTree:
    def __init__(self, dim=1):
        if dim >= 1:
            self.k = dim
            self.root = None
        else:
            raise Exception("Dimension is < 1")

    def __getitem__(self, pos):
        cnt = 0
        p = self.root
        while p is not None:
            if cnt == self.k:
                cnt = 0
            if p.item == pos:
                return pos
            elif p.item[cnt] >= pos[cnt]:
                p = p.left
            else:
                p = p.right
            
            cnt += 1
        raise Exception("No such element")

    def __setitem__(self, key, value):
        cnt = 0  
        p = self.root
        parent = p;
        if p is None:
            self.root = Node(value, None)
        while p is not None:
            if cnt == self.k:
                cnt = 0
            if p.item[cnt] >= value[cnt]:
                parent = p
                p = p.left
                if p is None:
                    parent.left = Node(value, parent)
            else:
                parent = p
                p = p.right
                if p is None:
                    parent.right = Node(value, parent)
            
            cnt += 1

class Node:
    def __init__(self, _item, _parent=None):
        self.parent = _parent
        self.item = _item
        self.left = None
        self.right = None

