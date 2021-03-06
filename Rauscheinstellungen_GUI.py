from tkinter import *
#import 
import math
import os
from numpy.core.function_base import linspace
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pywt
import scipy.stats as stats
from scipy.stats import linregress
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
# Websocket
import sys
import websocket
from functools import reduce
import struct
import signal
from datetime import datetime
import time

# GUI
root = Tk()
root.title("Kalibrierungstool")

# s: Referenzsignal
# ns: Nutzsignal
s_aufgenommen = False
ns_aufgenommen = False

sigma = 2 # Stärke Rauschreduzierung
dauer_aufnahme = 5 # In Sekunden
slope = 0 # Steigung der Regressionsgerade
intercept = 0 # Steigung der Regressionsgerade

sample_size = IntVar() # Slice Size
sample_size.set(7176)
n_splits = 7  # Anzahl der Teilungsschritte
plot_size = 10000 # Größe des dargestellten Ausschnitts im Plot
number_samples = 0
prozent_eingespart = 0
kanal = IntVar() # Betrachteter Kanal
kanal.set(0)

signal_input = [[],[],[],[]] # 4 Channels

Slopes = []
Intercepts = []
Stds = []

# Berechnung der Frequenzen der Subbänder mit Startfrequenz von 48000 Hertz
List = []
List_frequencies = []
value = 48000
for i in range(12):
    value = value/2.0
    List.append(value)
    List_frequencies.append(value*0.75)


# WebSocket Modul (einfach übernommen, da es so funktioniert hat)
zaehler = 0

def save(signal, frame):
    now = datetime.now()
    # print("Record stopped: ", now)
    sys.exit(0)

arg = sys.argv
# if len(arg) != 2:
#     print("Wrong usage: python3 " + arg[0] +
#           " [Socket Url: e.g. ws://10.0.0.1:8888]")
#     sys.exit()

# SOCKET = 'ws://10.0.0.107:8888'

SOCKET = arg[1]

FIRST_DATA_BYTE = 16
SENDING_INTERVAL_FACTOR = 1
COMPRESSION = "raw"

class DataView:
    def __init__(self, array, bytes_per_element=1):
        """
        bytes_per_element is the size of each element in bytes.
        By default we are assume the array is one byte per element.
        """
        self.array = array
        self.bytes_per_element = 1

    def __get_binary(self, start_index, byte_count, signed=False):
        integers = [self.array[start_index + x] for x in range(byte_count)]
        bytes = [integer.to_bytes(
            self.bytes_per_element, byteorder='big', signed=signed) for integer in integers]
        return reduce(lambda a, b: a + b, bytes)

    def get_bigint_64(self, start_index):
        bytes_to_read = 8
        return int.from_bytes(self.__get_binary(start_index, bytes_to_read), byteorder='big')

    def get_uint_16(self, start_index):
        bytes_to_read = 2
        return int.from_bytes(self.__get_binary(start_index, bytes_to_read), byteorder='big')

    def get_uint_8(self, start_index):
        bytes_to_read = 1
        return int.from_bytes(self.__get_binary(start_index, bytes_to_read), byteorder='big')

    def get_float_32(self, start_index):
        bytes_to_read = 4
        binary = self.__get_binary(start_index, bytes_to_read)
        return struct.unpack('>f', binary)[0]  # >f for big endian

    def length(self):
        return len(self.array)


def parse(byte_array):

    d = DataView(list(byte_array))

    max = d.length() if (4 % 2) == 0 else d.length() - 4

    res = {
        "value": d.get_uint_8(0),
        "request ID": d.get_uint_8(1),
        "number of records": d.get_uint_16(2),
        "number of sensors": d.get_uint_8(4),
        "number of errors": d.get_uint_16(6),
        "timestamp": d.get_bigint_64(8),
    }

    rec_count = res["number of records"]
    sen_count = res["number of sensors"]
    if (rec_count > 0):
        for index, value in enumerate(range(1, 5)):
            res[f"sensor{value}"] = [d.get_float_32(
                x) for x in range(FIRST_DATA_BYTE+index*4, max, sen_count*4)]
    return res

def on_open(ws):
    ws.send(json.dumps({
        "jsonrpc": "2.0",
        "method": "subscribeLiveData",
        "params": {
            "sendingIntervalFactor": SENDING_INTERVAL_FACTOR,
            "compression": COMPRESSION,
            "sensors": None},
        "id": 1}
    ))
    now = datetime.now()
    # print(f"Record started: {now}")

def on_message(ws, message):
    datei = open('daten.csv','w+')
    datei.write('Inhalt\n') 
    datei.close()

    global zaehler
    global dauer_aufnahme
    # print('Nachricht: ' + str(zaehler))
    zaehler += 1
    if (zaehler > dauer_aufnahme):
        ws.close()

    if isinstance(message, bytes):
        result = parse(message)
        global signal_input
        signal_input[0] = signal_input[0] + result[f"sensor1"]
        signal_input[1] = signal_input[1] + result[f"sensor2"]
        signal_input[2] = signal_input[2] + result[f"sensor3"]
        signal_input[3] = signal_input[3] + result[f"sensor4"]

    else:
        # print('Keine Nachricht mit Daten')
        if json.loads(message)['error']:
            print(f'Error: {json.loads(message)["error"]}')
            sys.exit(0)
# Websocket Modul Ende



# Zeigt Plot mit aktuellen Parametern an
def button_update_press():
    global plot_size
    plot_size = regler_plotsize.get()

    global ns_aufgenommen
    update_new_setting(ns_aufgenommen)
    
    update_plot(plot_size, ns_aufgenommen)

# Führt Kompression der Daten mit aktuellen Parametern durch
def update_new_setting(mit_ns):
    # Compress the full signal. Update Plot with new settings
    global n_splits
    global prozent_eingespart
    global sigma
    sigma = float(eingabe_sigma.get())
    s_size = sample_size.get()
    global s
    global signal_compressed
    if (mit_ns):
        global ns
        global ns_compressed
        ns_compressed = ns.copy()

    signal_size = len(s)
    signal_compressed = s.copy()
    global number_samples
    number_samples = math.floor(signal_size / s_size)

    global Slopes
    global Intercepts
    global Stds
    Slopes = []
    Intercepts = []
    Stds = []

    for number_sample in range(number_samples):
        signal_sample = s[(number_sample*s_size):((number_sample*s_size)+s_size)]
        
        # Komprimierung:

        # Make Splits
        if (n_splits > pywt.dwt_max_level(len(signal_sample), 'db5')):
            n_splits = pywt.dwt_max_level(len(signal_sample), 'db5')

        Subbands = pywt.wavedec(signal_sample, 'db5', level=n_splits)
        LowerSubbands = Subbands[1:]
        LowerSubbands.reverse()

        # Standardabweichungen berechnen
        ListSubB_Stds = []
        for i in range(n_splits):
            std = np.std(LowerSubbands[i], ddof=1)
            ListSubB_Stds.append(std)
        Stds.append(ListSubB_Stds)

        # Regressions Gerade bestimmen
        frequencies = List_frequencies[0:n_splits]
        slope_sample, intercept_sample, r, p, std = linregress(frequencies, ListSubB_Stds)
        Slopes.append(slope_sample)
        Intercepts.append(intercept_sample)

        # Treshold Werte bestimmen
        Thresholds = [[],[]] # B_low ; B_up
        for i in range(n_splits):
            reg_std = slope_sample*i + intercept_sample
            low = 0 - sigma * reg_std
            up = 0 + sigma * reg_std
            Thresholds[0].append(low)
            Thresholds[1].append(up)

        null_gesetzt = 0
        nicht_null_gesetzt = LowerSubbands[n_splits-1].size

        # Subbänder komprimieren
        ListSubB_K = LowerSubbands
        for i in range(n_splits):
            for value in range(LowerSubbands[i].size):
                # B
                threshold_low = Thresholds[0][i]
                threshold_up = Thresholds[1][i]
                if (LowerSubbands[i][value] > threshold_low and LowerSubbands[i][value] < threshold_up):
                    ListSubB_K[i][value] = 0
                    null_gesetzt += 1
                else:
                    nicht_null_gesetzt += 1
        prozent_eingespart = null_gesetzt / (null_gesetzt+nicht_null_gesetzt)

        # Rekonstruktion
        ListSubB_K.reverse()
        Subbands_rec = []
        Subbands_rec.append(Subbands[0])
        for i in range(n_splits):
            Subbands_rec.append(ListSubB_K[i])
        
        signalRec = pywt.waverec(Subbands_rec, 'db5')

        prozent_eingespart = prozent_eingespart*100
        prozent_eingespart = round(prozent_eingespart, 2)

        l3 = Label(root, text="Data Reduction: " + str(prozent_eingespart) + "%   ", font='Helvetica 12 bold')
        l3.place(x=900, y=150)

        for j in range(s_size-1):
            signal_compressed[(number_sample*s_size)+j] = signalRec[j]
        
    global slope
    global intercept
    slope = np.mean(Slopes)
    intercept = np.mean(Intercepts)

    # Nutzsignal komprimieren
    if (mit_ns):

        for number_sample in range(number_samples):
            signal_sample = ns[(number_sample*s_size):((number_sample*s_size)+s_size)]
        
            # Komprimierung:

            # Make Splits
            if (n_splits > pywt.dwt_max_level(len(signal_sample), 'db5')):
                n_splits = pywt.dwt_max_level(len(signal_sample), 'db5')

            Subbands = pywt.wavedec(signal_sample, 'db5', level=n_splits)
            LowerSubbands = Subbands[1:]
            LowerSubbands.reverse()

            # Treshold Werte bestimmen
            Thresholds = [[],[]] # B_low ; B_up
            for i in range(n_splits):
                reg_std = slope*i + intercept
                low = 0 - sigma * reg_std
                up = 0 + sigma * reg_std
                Thresholds[0].append(low)
                Thresholds[1].append(up)

            # Subbänder komprimieren
            ListSubB_K = LowerSubbands
            for i in range(n_splits):
                for value in range(LowerSubbands[i].size):
                    # B
                    threshold_low = Thresholds[0][i]
                    threshold_up = Thresholds[1][i]
                    if (LowerSubbands[i][value] > threshold_low and LowerSubbands[i][value] < threshold_up):
                        ListSubB_K[i][value] = 0

            # Rekonstruktion
            ListSubB_K.reverse()
            Subbands_rec = []
            Subbands_rec.append(Subbands[0])
            for i in range(n_splits):
                Subbands_rec.append(ListSubB_K[i])
        
            signalRec = pywt.waverec(Subbands_rec, 'db5')

            for j in range(s_size-1):
                ns_compressed[(number_sample*s_size)+j] = signalRec[j]
    
# Erzeugt Plot der Signale
def update_plot(plot_size, b):
    # Test if plot_size is to big
    global number_samples
    compressed_signal_size = number_samples * sample_size.get()
    if (plot_size > compressed_signal_size):
        plot_size = compressed_signal_size

    # Update Plot with given size
    signal_sample = s[0:plot_size]
    signal_sample_C = signal_compressed[0:plot_size]

    if (b):
        ns_sample = ns[0:plot_size]
        ns_sample_C = ns_compressed[0:plot_size]

    # Plot Original
    fig = Figure(figsize = (6, 3), facecolor = "white")
    plt = fig.add_subplot(111)
    plt.plot(range(len(signal_sample_C)),signal_sample_C, color='white')
    plt.plot(range(len(signal_sample)),signal_sample)
    canvas = FigureCanvasTkAgg(fig, master = root)
    canvas._tkcanvas.place(x=0,y=200)

    # Plot Komprimiert
    fig2 = Figure(figsize = (6, 3), facecolor = "white")
    plt = fig2.add_subplot(111)
    plt.plot(range(len(signal_sample)),signal_sample, color='white')
    plt.plot(range(len(signal_sample_C)),signal_sample_C)
    canvas2 = FigureCanvasTkAgg(fig2, master = root)
    canvas2._tkcanvas.place(x=600,y=200)

    if (b):
        # Plot Original Signal
        fig3 = Figure(figsize = (6, 3), facecolor = "white")
        plt = fig3.add_subplot(111)
        plt.plot(range(len(ns_sample_C)),ns_sample_C, color='white')
        plt.plot(range(len(ns_sample)),ns_sample)
        canvas3 = FigureCanvasTkAgg(fig3, master = root)
        canvas3._tkcanvas.place(x=0,y=500)

        # Plot Komprimiert Signal
        fig4 = Figure(figsize = (6, 3), facecolor = "white")
        plt = fig4.add_subplot(111)
        plt.plot(range(len(ns_sample)),ns_sample, color='white')
        plt.plot(range(len(ns_sample_C)),ns_sample_C)
        canvas4 = FigureCanvasTkAgg(fig4, master = root)
        canvas4._tkcanvas.place(x=600,y=500)

    l1 = Label(root, text="Original:")
    l1.place(x=300, y=210)

    l2 = Label(root, text="Compressed:")
    l2.place(x=900, y=210)

    l10 = Label(root, text="Reference Signal:")
    l10.place(x=30, y=210)

    l20 = Label(root, text="Signal:")
    l20.place(x=30, y=505)


# Einlesen der Daten vom Timeswipe
def button_aufnahme_press():
    aufnahme(False)

# Aufnahme Nutzsignal
def button_aufnahme_signal_press():
    global s_aufgenommen
    if (s_aufgenommen):
        aufnahme(True)
    else:
        print('First a noise model needs to be calculated from a reference signal')  

def aufnahme(s_aufg):
    global kanal
    global dauer_aufnahme 
    dauer_aufnahme = regler_dauer.get()
    global signal_input
    global zaehler
    signal_input = [[],[],[],[]]
    zaehler = 0

    # Vom Websocket Modul
    signal.signal(signal.SIGINT, save)
    ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_message=on_message)
    ws.run_forever()

    if (s_aufg == False):

        # Eingelesenes Signal
        global s 
        s = signal_input[kanal.get()]
        s = s[0:dauer_aufnahme*48000]

        global s_aufgenommen
        s_aufgenommen = True

        button_update_press()

    if (s_aufg == True):
 
        # Eingelesenes Signal
        global ns
        ns = signal_input[kanal.get()]
        ns = s[0:dauer_aufnahme*48000]

        global ns_aufgenommen
        ns_aufgenommen = True

        button_update_press()

# Speichert die aktuellen Parameter in JSON Datei
def button_speichern_press():
    x = {
        "Sigma": sigma,
        "Slice_size": sample_size.get(),
        "Slope": slope,
        "Intercept": intercept
    }

    with open('Calibration/calibration.json', 'w') as fp:
        json.dump(x, fp)

# Öffnet Fenster mit der Regressionsgerade
def button_gerade_press():
    global n_splits
    global slope
    global intercept
    global Stds

    fenster_gerade = Toplevel(root)
    fenster_gerade.title("Regression Curve")

    # Plot
    fig_g = Figure(figsize = (6, 4), facecolor = "white")
    plt = fig_g.add_subplot(111)

    frequencies = List_frequencies[0:n_splits]
    
    for i in range(number_samples):
        plt.scatter(frequencies, Stds[i])  

    X = np.linspace(0,18000,10)
    Y = slope*X+intercept
    plt.set_ylabel('Standard Deviation of Slices')
    plt.set_xlabel('Frequency')
    plt.plot(X, Y, color = 'black', label = 'Regressions Curve')
    plt.legend()

    canvas = FigureCanvasTkAgg(fig_g, master = fenster_gerade)
    canvas._tkcanvas.grid(row=0,column=0, pady = 50)



button_aufnahme_reference = Button(root, text='Record reference signal', width=40, command=button_aufnahme_press)
button_aufnahme_reference.place(x=155, y=15)

button_aufnahme_signal = Button(root, text='Record signal', width=40, command=button_aufnahme_signal_press)
button_aufnahme_signal.place(x=155, y=45)

regler_dauer = Scale(root, from_=0, to=10, orient=HORIZONTAL)
regler_dauer.place(x=550, y=20)
regler_dauer.set(2)
dauer_aufnahme = regler_dauer.get()

label_dauer = Label(root, text="Duration of Recording (s)")
label_dauer.place(x=550, y=60)

label_komprimierungseinstellungen = Label(root, text="Compression settings:")
label_komprimierungseinstellungen.place(x=120, y=78)

#regler_sigma = Scale(root, from_=1, to=6, orient=HORIZONTAL)
#regler_sigma.place(x=130, y=115)
#regler_sigma.set(2)

eingabe_sigma = StringVar()
eingabefeld_sigma = Entry(root, width = 10 ,textvariable=eingabe_sigma)
eingabefeld_sigma.place(x=130, y=105)
eingabe_sigma.set('2')

label_sigma = Label(root, text="Sigma")
label_sigma.place(x=130, y=125)

button_speichern = Button(root, text='Save', width=35, command=button_speichern_press)
button_speichern.place(x=890,y=90)

button_gerade_anzeigen = Button(root, text='Show Regression Curve', width=35, command=button_gerade_press)
button_gerade_anzeigen.place(x=890,y=40)

regler_plotsize = Scale(root, from_=1000, to=100000, orient=HORIZONTAL)
regler_plotsize.place(x=500, y=120)
regler_plotsize.set(10000)

label_plotsize = Label(root, text="Size of plottet window")
label_plotsize.place(x=500, y=160)

button_update = Button(root, text='Update Plot', command=button_update_press, bg = "grey")
button_update.place(x=132, y=160)

# Plot Original
fig = Figure(figsize = (6, 3), facecolor = "white")
plt = fig.add_subplot(111)
plt.plot(0,0)
#plt.plot(range(data_sample.size),data_sample)
canvas = FigureCanvasTkAgg(fig, master = root)
# canvas._tkcanvas.grid(row=3,column=0, pady = 135)
canvas._tkcanvas.place(x=0,y=200)

# Plot Komprimiert
fig2 = Figure(figsize = (6, 3), facecolor = "white")
plt = fig2.add_subplot(111)
plt.plot(0,0)
canvas2 = FigureCanvasTkAgg(fig2, master = root)
# canvas2._tkcanvas.grid(row=3,column=1, pady = 135)
canvas2._tkcanvas.place(x=600,y=200)

# Plot Original Signal
fig3 = Figure(figsize = (6, 3), facecolor = "white")
plt = fig3.add_subplot(111)
plt.plot(0,0)
#plt.plot(range(data_sample.size),data_sample)
canvas3 = FigureCanvasTkAgg(fig3, master = root)
canvas3._tkcanvas.place(x=0,y=500)

# Plot Komprimiert Signal
fig4 = Figure(figsize = (6, 3), facecolor = "white")
plt = fig4.add_subplot(111)
plt.plot(0,0)
canvas4 = FigureCanvasTkAgg(fig4, master = root)
canvas4._tkcanvas.place(x=600,y=500)

l1 = Label(root, text="Original:")
l1.place(x=300, y=210)

l2 = Label(root, text="Compressed:")
l2.place(x=900, y=210)

l10 = Label(root, text="Reference Signal:")
l10.place(x=30, y=210)

l20 = Label(root, text="Signal:")
l20.place(x=30, y=505)

l3 = Label(root, text="Data Reduction: " + str(prozent_eingespart) + "%", font='Helvetica 12 bold')
l3.place(x=900, y=150)

# Sample Size Radiobottons
def on_click():
    global n_splits
    if (sample_size.get() == 48000):
        n_splits = pywt.dwt_max_level(48000, 'db5')-2
    elif (sample_size.get() == 24000):
        n_splits = pywt.dwt_max_level(24000, 'db5')-1
    elif (sample_size.get() == 12000):
        n_splits = pywt.dwt_max_level(12000, 'db5')
    elif (sample_size.get() == 6000):
        n_splits = pywt.dwt_max_level(6000, 'db5')

r1 = Radiobutton(root, text="1 s", padx = 20, variable=sample_size, command=on_click, value=48000)
r1.place(x=265, y=100)
r2 = Radiobutton(root, text="500 ms", padx = 20, variable=sample_size, command=on_click, value=24000)
r2.place(x=265, y=120)
r3 = Radiobutton(root, text="250 ms", padx = 20, variable=sample_size, command=on_click, value=12000)
r3.place(x=265, y=140)
r4 = Radiobutton(root, text="125 ms", padx = 20, variable=sample_size, command=on_click, value=6000)
r4.place(x=265, y=160)
r2.select()

# Kanal Radiobottons
def on_click_kanal():
    global s 
    global kanal
    global signal_input
    s = signal_input[kanal.get()]
    s = s[0:dauer_aufnahme*48000]

rd1 = Radiobutton(root, text="1", padx = 20, variable=kanal, command=on_click_kanal, value=0)
rd1.place(x=750, y=45)
rd2 = Radiobutton(root, text="2", padx = 20, variable=kanal, command=on_click_kanal, value=1)
rd2.place(x=750, y=65)
rd3 = Radiobutton(root, text="3", padx = 20, variable=kanal, command=on_click_kanal, value=2)
rd3.place(x=750, y=85)
rd4 = Radiobutton(root, text="4", padx = 20, variable=kanal, command=on_click_kanal, value=3)
rd4.place(x=750, y=105)
rd1.select()

label_kanal = Label(root, text="Channel:")
label_kanal.place(x=755, y=23)

label_sample_size = Label(root, text="Slice Size:")
label_sample_size.place(x=280, y=78) # 123

root.geometry('1200x810')

root.mainloop()