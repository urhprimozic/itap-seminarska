# prebarvanje reference 18 v prave barve
import utils
import matplotlib.pylab as plt

N = 18

ref4d = utils.reshape_reference(utils.get_reference(N))
plt.figure(figsize=(8, 8))
plt.imshow(utils.reshape_reference(utils.get_reference(N)), cmap=utils.category_cmap, norm=utils.category_norm)
plt.savefig('../img/brnik_ref.png')

data4d = utils.data_to_4D(utils.get_data(N))
plt.figure(figsize=(8, 8))
plt.imshow(data4d[..., 8, [3,2,1]] * 3.5)
plt.savefig('../img/brnik_data.png')