import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_excel("algores.xlsx")
plt.plot(df["Vertices"], df.iloc[:, 1], color="red", label="E ~ V")
plt.plot(df["Vertices"], df.iloc[:, 2], color="black", label='E ~ V^1.5')
plt.plot(df["Vertices"], df.iloc[:, 3], color="blue", label="E ~ V^2")
plt.xlabel("Vertices number")
plt.ylabel("Time (us)")
plt.legend()
plt.show()
