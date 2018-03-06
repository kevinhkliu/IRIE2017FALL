#################################################
#Entity Class
#################################################
class Character:
    def __init__(self, identify):
        self.identify = identify
        self.sibling = []
        self.grand = []
        self.gender = -1
        self.main_role = False
        self.slave = False
        self.father = None
        self.mother = None
        self.freq = 0
        self.l_freq = 0