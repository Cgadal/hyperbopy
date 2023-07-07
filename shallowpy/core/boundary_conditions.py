class BoundaryCondition:

    def __init__(self, BC, side):
        self.BC = BC  # constant or 'symmetry'
        self.side = side  # 'left' or 'right'

    def update_bc(self, w):
        if self.BC == 'symmetry':
            if self.side == 'left':
                w[0] = w[1]
            if self.side == 'right':
                w[-1] = w[-2]
        elif type(self.BC) == int or type(self.BC) == float:
            if self.side == 'left':
                w[0] = self.BC
            if self.side == 'right':
                w[-1] = self.BC


class BoundaryConditions:

    def __init__(self, BCs):
        self.BCs = BCs
        self.parse_BCs()

    def parse_BCs(self):
        self.BCS_parsed = [[BoundaryCondition(BC, 'left' if i == 0 else 'right')
                            for i, BC in enumerate(BC_var)] for BC_var in self.BCs]

    def update_all(self, W):
        for w, BC_var in zip(W, self.BCS_parsed):
            for BC in BC_var:
                BC.update_bc(w)
