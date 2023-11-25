import random
a = [0.345,0.655]
b = [0.122,0.878]
a = [a[i]+b[i] for i in range(len(a))]
a=[a[i]/2 for i in range(len(a))]
print(sum(a))
    