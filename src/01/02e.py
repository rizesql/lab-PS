import numpy as np
import matplotlib.pyplot as plt

e = np.random.rand(128, 128)

plt.imshow(e)

plt.tight_layout()
plt.savefig("src/01/02e.pdf", format="pdf")
plt.show()
