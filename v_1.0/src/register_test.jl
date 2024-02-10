using Bloqade
using PythonCall
plt = pyimport("matplotlib.pyplot")

data = [0.4, 0.3]

print(data[1])

amp = [sqrt(data[1]), sqrt(1-data[1]), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

reg = ArrayReg(amp)