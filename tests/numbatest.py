import numpy as np
from matplotlib import pyplot as plt

from brainforge.ops import ConvolutionOp as NPConv
from brainforge.numbaops.lltensor import ConvolutionOp as NBConv

npop = NPConv()
nbop = NBConv()
A = np.random.uniform(0., 1., (1, 1, 12, 12))
F = np.random.uniform(0., 1., (4, 4, 1, 1))

npO = npop.apply(A, F, mode="valid")
nbO = nbop.apply(A, F, mode="valid")

d = np.abs(npO - nbO)

vmax, vmin = max(npO.max(), nbO.max()), min(npO.min(), nbO.min())
fig, axarr = plt.subplots(2, 2)
axarr[0][0].imshow(A[0, 0], vmin=0, vmax=1, cmap="autumn")
axarr[0][0].set_title("A")
axarr[0][1].imshow(d[0, 0], cmap="seismic")
axarr[0][1].set_title("d")
axarr[1][0].imshow(npO[0, 0], vmin=vmin, vmax=vmax, cmap="hot")
axarr[1][0].set_title("npO")
axarr[1][1].imshow(nbO[0, 0], vmin=vmin, vmax=vmax, cmap="hot")
axarr[1][1].set_title("nbO")
plt.show()
print("DIFF: {}".format(np.abs(npO - nbO).mean()))
