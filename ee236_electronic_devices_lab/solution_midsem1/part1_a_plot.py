import matplotlib.pyplot as plt

temp = [20, 30, 40, 50, 60, 70, 80]
v_d_pn = [0.655, 0.637, 0.618, 0.600, 0.582, 0.563, 0.544]
v_d_zen = [-0.556, -0.534, -0.512, -0.490, -0.468, -0.445, -0.422]

plt.plot(temp, v_d_pn, label="I_d = 2mA")
plt.legend()
plt.savefig("v_d_pn.png")
plt.show()

plt.plot(temp, v_d_zen, label="I_d = -0.5mA")
plt.legend()
plt.savefig("v_d_zen.png")
plt.show()