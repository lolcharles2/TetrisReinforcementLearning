class A:
    def __init__(self):
        self.num = 1

    def ad(self, x):
        x += 1

    def test(self):
        self.ad(self.num)

ob = A()
ob.test()
print(ob.num)