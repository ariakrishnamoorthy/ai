FIELD_ELEMENTS = [
    [1, 0], [2, 0], [0, 1], [0, 2],
    [1, 1], [1, 2], [2, 1], [2, 2]
]

def multiply(mod, elem1, elem2):
    total = [0] * (len(elem1) + len(elem2)-1)

    k = 0
    for i in range(len(elem1)):
        k = i
        for j in range(len(elem2)):
            total[k] += elem1[i] * elem2[j]
            k += 1
    for i in range(len(total)):
        total[i] = total[i] % mod
    return total

def add(mod, elem1, elem2):
    elem1 = list(elem1)
    elem2 = list(elem2)
    if (len(elem1) < len(elem2)):
        padding = [0] * (len(elem2) - len(elem1))
        elem1.extend(padding)
    if (len(elem2) < len(elem1)):
        padding = [0] * (len(elem1) - len(elem2))
        elem2.extend(padding)
    
    added = []
    for i in range(0, len(elem1)):
        added.append(elem1[i] + elem2[i])
        added[i] = added[i] % mod
    return added

def reduce_field(mod, maxdeg, equal, elem1, elem2):
    total = multiply(mod, elem1, elem2)
    while len(total) > maxdeg + 1:
        if total[-1] == 0:
            total.pop()
            continue

        lead_coef = total[-1]
        diff = len(total) - maxdeg -2
        diff_arr = [0] * diff 
        diff_arr.append(lead_coef)
        newelem = multiply(mod, diff_arr, equal)
        total = add(mod, total, newelem)
        total.pop()
    while len(total) < 2:
        total.append(0)
    return total

