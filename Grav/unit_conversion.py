import numpy as np

G = 6.67e-11

SEC_TO_MEGAYRS = 3.17e-14

KG_TO_MSOL = 5e-31

M_TO_PARSEC = 3.24078e-17

print(G * (M_TO_PARSEC ** 3) / ((KG_TO_MSOL) * (SEC_TO_MEGAYRS ** 2)))
