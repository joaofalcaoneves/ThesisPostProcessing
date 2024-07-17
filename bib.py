def time_courant(maxCo=1, minCellVol=1, maxVel=1):
    maxDeltaT = maxCo * minCellVol / maxVel
    return maxDeltaT

a = [0.100348]
b = []
for i in a:
    print(i)
    b.append(time_courant(maxCo=0.3, minCellVol=i, maxVel=0.495))

print(b)