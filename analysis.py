import matplotlib.pyplot as plt
import numpy as np

# Dados de calibração das câmeras
cameras = ['GoPro', 'iPhone']
focal_lengths = {
    'GoPro': [2454.856, 2467.604],
    'iPhone': [1613.976, 1620.017]
}
principal_points = {
    'GoPro': [2612.995, 1916.762],
    'iPhone': [2011.360, 1500.727]
}
dist_coeffs = {
    'GoPro': [-0.3279, 0.1941, 0.0031, -0.0020, -0.0805],
    'iPhone': [0.0595, -0.1370, 0.00005, 0.00092, 0.0856]
}

# Criar gráficos das distâncias focais
plt.figure(figsize=(12, 10))

# Gráfico das distâncias focais

bar_width = 0.35
x = np.arange(len(cameras))

plt.bar(x - bar_width/2, [focal_lengths['GoPro'][0], focal_lengths['GoPro'][1]], width=bar_width, label='GoPro')
plt.bar(x + bar_width/2, [focal_lengths['iPhone'][0], focal_lengths['iPhone'][1]], width=bar_width, label='iPhone')

plt.xticks(x, ['f_x', 'f_y'])
plt.ylabel('Valor (pixels)')
plt.title('Distâncias Focais')
plt.legend()
plt.grid()

# Gráfico dos pontos principais
plt.figure(figsize=(12, 10))
plt.bar(x - bar_width/2, [principal_points['GoPro'][0], principal_points['GoPro'][1]], width=bar_width, label='GoPro')
plt.bar(x + bar_width/2, [principal_points['iPhone'][0], principal_points['iPhone'][1]], width=bar_width, label='iPhone')

plt.xticks(x, ['c_x', 'c_y'])
plt.ylabel('Valor (pixels)')
plt.title('Pontos Principais')
plt.legend()


plt.tight_layout()
plt.grid()
plt.show()

# Gráfico dos coeficientes de distorção
plt.figure(figsize=(12, 4))
x = np.arange(len(dist_coeffs['GoPro']))

plt.bar(x - 0.2, dist_coeffs['GoPro'], width=0.4, label='GoPro')
plt.bar(x + 0.2, dist_coeffs['iPhone'], width=0.4, label='iPhone')

plt.xticks(x, ['k_1', 'k_2', 'p_1', 'p_2', 'k_3'])
plt.ylabel('Valor')
plt.title('Coeficientes de Distorção')
plt.grid()
plt.legend()
plt.show()
