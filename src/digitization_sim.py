import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from pyqtgraph import examples
from pyqtgraph.Qt import QtCore
from scipy import fft

def basic_sim():
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
    
def real_time_plot_sim():
    sf = 44000
    
    t_period = 0.1
    t_step = 1/sf
    t = np.linspace(0, t_period, int(t_period*sf), endpoint=False)
    f = fft.fftfreq(len(t), t_step)[:len(t)//2]
    
    f_x_min = 20
    f_x_max = 200
    t_0 = t[len(t)//2]
    audio_sim = [np.sinc(2*np.pi*f_x*(t - t_0)) for f_x in range(f_x_min, f_x_max)]
    
    audio_input = np.array(audio_sim)
    
    # audio_input_fft = fft.rfft(audio_input)[:len(f)]
    # audio_input_fft_mag = np.abs(audio_input_fft)
    
    # plt.figure()
    # plt.plot(t, audio_input)
    # plt.show()
    
    # plt.figure()
    # plt.plot(f, audio_input_fft_mag)
    # plt.show()
    
    pg.mkQApp("Plotting Example")
    #mw = QtWidgets.QMainWindow()
    #mw.resize(800,800)

    win = pg.GraphicsLayoutWidget(show=True, title="Basic plotting examples")
    win.resize(1000,600)
    win.setWindowTitle('pyqtgraph example: Plotting')

    # Enable antialiasing for prettier plots
    pg.setConfigOptions(antialias=True)
    
    p6 = win.addPlot(title="Updating plot")
    curve = p6.plot(pen='y')
    # data = np.random.normal(size=(10,1000))
    ptr = [0]
    def update(p6=p6, curve=curve, data=audio_input, ptr=ptr):
        data = audio_input[ptr[0]%10, :]
        audio_input_fft = fft.rfft(data)[:len(f)]
        audio_input_fft_mag = np.abs(audio_input_fft)
        data = audio_input_fft_mag[:250]
        curve.setData(data)
        if ptr[0] == 0:
            p6.enableAutoRange('xy', False)  ## stop auto-scaling after the first data set is plotted
        ptr[0] += 1
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50)
    
    pg.exec()
    
def pyqtgraph_examples():
    examples.run()

if __name__ == "__main__":
    # basic_sim()
    real_time_plot_sim()
    # pyqtgraph_examples()
    