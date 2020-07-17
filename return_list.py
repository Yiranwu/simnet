

def func1():
    return {'1':1},2

def func2():
    return func1(),3

(ret1,ret2), ret3 = func2()
print(ret1, ret2, ret3)