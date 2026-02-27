from src.spin import *

I = Spin(7/2)
print("Initial state:", I.state)
print("Expectation values: <Ix> =", I.Ix(), ", <Iy> =", I.Iy(), ", <Iz> =", I.Iz())


