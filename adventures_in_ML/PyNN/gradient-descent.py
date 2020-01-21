x_old = 0 # The value does not matter as long as abs(x_new - x_old) > precision
x_new = 6 # The algorithm starts at x=6
gamma = 0.01 # step size
precision = 0.00001

def df(x):
    y = 4 * x**3 - 9 * x**2
    return y

step = 0
while abs(x_new - x_old) > precision:
    step+=1
    print("Step "+str(step)+" : x_old=" +str(round(x_old,4))+"\t    x_new=" +str(round(x_new,4)))
    x_old = x_new
    _df = df(x_old)
    x_new += -gamma * _df
    print("\t   df="+str(_df))

print("The local minimum occurs at %f" % x_new)