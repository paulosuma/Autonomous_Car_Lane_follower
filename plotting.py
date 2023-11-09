import matplotlib.pyplot as plt
import numpy as np
from waypoint_list import WayPoints



# velocity = np.loadtxt("./src/mp2/src/text.txt", dtype=float)
acceleration = np.loadtxt("./src/mp2/src/accel.txt", dtype=float)
xandy_positions = np.loadtxt("./src/mp2/src/xandypos.txt", dtype=float)
x, y = xandy_positions[:, 0], xandy_positions[:, 1]
waypoints = WayPoints()
waylist = np.array(waypoints.getWayPoints())
wayplot = []
for i in range(0, len(waylist), 10):
        wayplot.append(waylist[i])
wayplot[-1] = waylist[-1]
wayplot = np.array(wayplot)
# print(wayplot[:, 1])
# print(waylist)


plt.plot(np.linspace(0, 125, len(acceleration)), acceleration)
plt.title("plot of accelaration over time (s)")
# plt.plot(x,y, "o")
# plt.plot(waylist[:, 0], waylist[:, 1], '*r')
# plt.plot(wayplot[:, 0], wayplot[:, 1], 'dy')
# plt.title("plot of robot trajectory over time")
# plt.xlabel("x_position")
# plt.ylabel("y_position")
plt.show()