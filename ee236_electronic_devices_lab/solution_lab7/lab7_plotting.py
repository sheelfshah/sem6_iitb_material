import matplotlib.pyplot as plt

# # exp1
# v_z = [0.73, 0.69, 0, -0.99, -1.99, -3, -3.99, -4.99, -5.6, -5.67]
# i_z = [1.8, 0.52, 0, -0.1e-3, -0.2e-3, -0.3e-3, -0.4e-3, -1.8e-3, -0.596, -11.8]
# plt.plot(v_z, i_z)
# plt.xlabel("V_z")
# plt.ylabel("I_z (mA)")
# plt.title("IV characteristic of zener")
# plt.axhline(y=0, color='k', linestyle='--')
# plt.axvline(x=0, color='k', linestyle='--')
# plt.axvline(x=-5.67, color='r', linestyle='--')
# plt.savefig("exp1.png")
# plt.show()

# # exp2

# # part1

# v_gs = [0, 0.7, 1.4, 2.1, 2.8, 3.5, 4.3]
# i_d = [0, 0.012, .14, .25, .36, .44, .54]
# x = [0.01*i for i in range(400)]
# y = [0.183*a - 0.116 for a in x]
# plt.plot(v_gs, i_d)
# plt.plot(x, y)
# plt.xlabel("V_gs")
# plt.ylabel("I_d (mA)")
# plt.title("Id-Vgs characteristic of NMOS")
# plt.axhline(y=0, color='k', linestyle='--')
# plt.axvline(x=0, color='k', linestyle='--')
# plt.axvline(x=0.636, color='r', linestyle='--')
# plt.savefig("part1.png")
# plt.show()

# part 2
v_ds = [0, .1, .2, .4, .8, 1.2, 1.6, 1.8, 1.9, 2, 2.5, 3, 3.5, 4, 4.5, 5]
i_d_25 = [0, 1.62e-1, .31, .57, .98, 1.23, 1.33, 1.35, 1.35, 1.36, 1.373, 1.385, 1.396, 1.404, 1.414, 1.422]
i_d_30 = [0, 1.9e-1, .37, .71, 1.26, 1.65, 1.89, 1.96, 1.98, 1.99, 2.04, 2.06, 2.07, 2.1, 2.12, 2.13]
i_d_35 = [0, 2.3e-1, .45, .85, 1.54, 2.07, 2.47, 2.6, 2.66, 2.7, 2.83, 2.87, 2.9, 2.92, 2.93, 2.95]

plt.plot(v_ds, i_d_25, label = "Vgs = 2.5V")
plt.plot(v_ds, i_d_30, label = "Vgs = 3.0V")
plt.plot(v_ds, i_d_35, label = "Vgs = 3.5V")
plt.xlabel("V_ds")
plt.ylabel("I_d (mA)")
plt.title("Id-Vds characteristic of NMOS")
plt.axhline(y=0, color='k', linestyle='--')
plt.axvline(x=0, color='k', linestyle='--')
plt.savefig("part2.png")
plt.legend()
plt.show()

# part 3
v_gs = [0, .2, .4, .5, .6, .7, .8, .9, 1, 1.2, 1.4, 1.7, 2, 2.5, 3, 3.5, 4, 4.5, 5]
i_d = [0, 0, 0, 5e-4, 3.7e-3, 1.21e-2, 2.81e-2, 5.02e-2, 8.22e-2, 1.64e-1, .278, .5, .78, 1.37, 2.06, 2.89, 3.81, 4.8, 5.89]
plt.plot(v_gs, i_d)
plt.xlabel("V_gs")
plt.ylabel("I_d (mA)")
plt.title("Id-Vgs characteristic of NMOS in saturation")
plt.axhline(y=0, color='k', linestyle='--')
plt.axvline(x=0, color='k', linestyle='--')
plt.savefig("part3.png")
plt.show()
