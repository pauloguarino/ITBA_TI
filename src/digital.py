import numpy as np
from scipy import fft
import matplotlib.pyplot as plt

def digitization_sim():
    t_start = 0
    t_stop = 2
    t_length = t_stop - t_start
    t_step = 1e-6
    t = np.linspace(t_start, t_stop, int(t_length/t_step), endpoint=False)
    f = fft.fftshift(fft.fftfreq(len(t), t_step))
    
    f_x = 20
    t_0 = t[len(t)//2]
    x = np.sinc(2*np.pi*f_x*(t - t_0))
    
    x_fft = fft.fftshift(fft.fft(x))
    x_fft_mag = np.abs(x_fft)
    
    sf = 20000
    subsampling = int((1/sf)//t_step)
    sf = 1/(subsampling*t_step)
    x_s = np.zeros(len(t))
    x_s[::subsampling] = x[::subsampling]
    
    x_s_fft = fft.fftshift(fft.fft(x_s))
    x_s_fft_mag = np.abs(x_s_fft)
    
    t_d = t[::subsampling]
    x_d = x[::subsampling]
    
    n_bits = 4 # supongo que la señal está normalizada en amplitud
    n_levels = 2**n_bits - 1
    levels = (np.arange(n_levels) - (n_levels - 1)/2)/n_levels
    x_q = np.zeros(len(t))
    for i in range(0, len(x_q), subsampling):
        x_q[i] = levels[np.argmin(np.abs(levels - x_s[i]))]
    
    x_q_fft = fft.fftshift(fft.fft(x_q))
    x_q_fft_mag = np.abs(x_q_fft)
    
    x_d_q = x_q[::subsampling]
    
    f_zoom_percent = 20
    f_zoom = slice(int(len(f)*((50 - f_zoom_percent/2)/100)), int(len(f)*((50 + f_zoom_percent/2)/100)))
    
    plt.figure()
    plt.plot(t, x)
    plt.show()
    
    plt.figure()
    plt.plot(f[f_zoom], x_fft_mag[f_zoom])
    plt.show()
    
    plt.figure()
    plt.plot(t, x_s)
    plt.show()
    
    plt.figure()
    plt.plot(f[f_zoom], x_s_fft_mag[f_zoom])
    plt.show()
    
    plt.figure()
    plt.plot(t_d, x_d)
    plt.show()
    
    plt.figure()
    plt.plot(t, x_q)
    plt.show()
    
    plt.figure()
    plt.plot(t_d, x_d_q)
    plt.show()
    
    plt.figure()
    plt.plot(f[f_zoom], x_q_fft_mag[f_zoom])
    plt.show()

if __name__ == "__main__":
    digitization_sim()
