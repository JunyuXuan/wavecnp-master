import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

bar_names = []
bar_1d = []
bar_2d = []


for line in open("results/levels1", "r"):
    l, d = line.split()
    bar_names.append(int(l))
    bar_1d.append(float(d))

for line in open("results/levels2", "r"):
    l, d2 = line.split()
    # bar_names.append(l)
    bar_2d.append(float(d2))

print(bar_names)
print(bar_1d)

fig, axs = plt.subplots(1, 2, figsize=(8, 3))


axs[0].plot(bar_names, bar_1d, 'o-', color='black', linewidth=2.0, markersize=12)
axs[0].set_title("1-D")
axs[0].set_xlabel("levels")
axs[0].set_ylabel("predictive log-likelihood")
axs[0].grid(axis='y')
axs[1].plot(bar_names, bar_2d, 'v-', color='black', linewidth=2.0, markersize=12)
axs[1].set_title('2-D')
axs[1].set_xlabel("levels")
axs[1].set_ylabel("predictive log-likelihood")
axs[1].grid(axis='y')
fig.tight_layout()
plt.show()
plt.draw()

fig.savefig('_experiments/figs/levels.pdf',bbox_inches='tight')

