import matplotlib.pyplot as plt

temp = [35, 45, 55, 65, 75]

ff_vals = [0.517, 0.493, 0.475, 0.453, 0.430]
plt.plot(temp, ff_vals)
plt.savefig("ff_v_temp.png")
plt.show()

vd_1 = [0.29, 0.26, 0.24, 0.21, 0.18]
vd_2 = [0.35, 0.32, 0.29, 0.27, 0.24]
vd_5 = [0.43, 0.40, 0.38, 0.35, 0.33]
plt.plot(temp, vd_1, label="I_d = 1mA")
plt.plot(temp, vd_2, label="I_d = 2mA")
plt.plot(temp, vd_5, label="I_d = 5mA")
plt.legend()
plt.savefig("vd_v_temp.png")
plt.show()

voc_vals = [0.392, 0.368, 0.343, 0.319, 0.294]
plt.plot(temp, voc_vals)
plt.savefig("voc_v_temp.png")
plt.show()

rs = [0, 10, 30]
rsh = [100, 500, 5000]

voc_rs = [0.42, 0.41, 0.39]
isc_rs = [8e-3, 7.9e-3, 7.6e-3]
ff_rs = [0.64, 0.54, 0.38]

plt.plot(rs, voc_rs, label="V_oc")
plt.savefig("voc_rs.png")
plt.show()

plt.plot(rs, isc_rs, label="I_sc")
plt.savefig("isc_rs.png")
plt.show()

plt.plot(rs, ff_rs, label="FF")
plt.savefig("ff_rs.png")
plt.show()

voc_rsh = [0.38, 0.41, 0.42]
isc_rsh = [7.2e-3, 7.8e-3, 8e-3]
ff_rsh = [0.41, 0.52, 0.54]

plt.plot(rsh, voc_rsh, label="V_oc")
plt.savefig("voc_rsh.png")
plt.show()

plt.plot(rsh, isc_rsh, label="I_sc")
plt.savefig("isc_rsh.png")
plt.show()

plt.plot(rsh, ff_rsh, label="FF")
plt.savefig("ff_rsh.png")
plt.show()