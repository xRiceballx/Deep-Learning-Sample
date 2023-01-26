# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks
from scipy.signal import argrelmax,argrelmin
import matplotlib.pyplot as plt
import os
import time


def fir_filter(x, taps, Fc, Fs, p_zero="bandpass", win="hamming"):
    u""" Filter
    @param  x   波形
    @param  taps    次数
    @param  Fc  カットオフ周波数 (~Fs/2 [Hz]),バンドパスの場合は[Fc_l, Fc_h]で指定
    @param  Fs  サンプリング周波数 (200 [Hz])
    @param  pass_zero {True, False, ‘bandpass’, ‘lowpass’, ‘highpass’, ‘bandstop’}
    @param  win {‘hamming’, ‘boxcar’}
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html
    """
    filter1 = signal.firwin(numtaps=taps, cutoff=Fc, fs=Fs, window=win,
                            pass_zero=p_zero)  # default:hamming / boxcar

    y = signal.filtfilt(b=filter1, a=1, x=x)  # filtfilt is zero-phase

    return y


def moving_average(x, num):
    u""" 移動平均
    @param  x   波形
    @param  num 移動平均個数
    """
    b = np.ones(num) / num  # 係数
    # np.ones(num)は、1をnum個並べた配列を作成します
    # np.ones(num)＝ array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

    y = np.convolve(x, b, mode='valid')
    # 移動平均を求めます
    # 基本的には配列xに配列bを1つづつ移動しながら掛け合わせていきます。
    # modeは掛け合わせる時の始点と終点の違いです。詳細は以下のURLをご確認ください。
    # mode='valid'の場合は、移動平均後の波形の長さは（-num+1）減少します。
    #　https://deepage.net/features/numpy-convolve.html

    return y




######
# main
######
if __name__ == "__main__":

    bcg_file = "sample2_bcg.csv"

    # BCGデータの読み込み
    bcg_data = pd.read_csv(bcg_file)
    bcg = bcg_data["0"].values

    # サンプリング周波数設定
    Fs = 125  # 125

    #　バンドパスフィルター
    ecg_bp = fir_filter(x=bcg, taps=15, Fc=[5,20], Fs=Fs, p_zero="bandpass")

    # # ロウパスの場合は以下のように書く
    # ecg_lp = fir_filter(x=bcg, taps=15, Fc=[20], Fs=Fs, p_zero="lowpass")

    # 移動平均
    bcg_ma = moving_average(ecg_bp, 5)

    """
    # plot
    plt.plot(bcg_ma)
    plt.show()
    """

#サンプルデータのグラフ、ピークの回数をヒストグラムで保存するメソッド    
def savegraphs(a,b,c,n):
    bcg = bcg[a:b]
    time = bcg["time"]
    wave = bcg["value"]
    
    time = np.array(time)
    wave = np.array(wave)
    
    peaks = argrelmax(wave, order=1)
    c = str(c)
    
    if n == 1:
        plt.plot(time, wave)
        plt.plot(time[peaks], wave[peaks], "x", label="peak_max")
        plt.xlabel("time")
        plt.ylabel("value")
        plt.legend(loc="upper right")
 
        #plt.savefig("10sec/" + "dotgraph" + c + ".png")
        plt.show()
        
        
    else:  
        #上のコードによって描かれたグラフの頂点を求め、Y軸の大きさと数でヒストグラムを準備する
        peakvalue = x[peaks]
        peakvalues = pd.DataFrame({'value':peakvalue})
        #print(peakvalues)
        
        Histgram, hist = plt.subplots()
        
        y = peakvalue
        hist.set_xlim(-20000,20000)
        hist.set_ylim(0,22)
        
        plt.hist(y, bins=50)
        plt.savefig("30sec/" + "histgram" + c + ".png")

    
#30秒間隔のヒストグラム表示
ran = 0
        
for i in range(3):
    savegraphs(ran+1, ran+125, i, 1)
    ran = ran + 125


#複数グラフを並べて表示する新規メソッド        
def drawgraphs(a,b):
    x = bcg[a:b]
    peaks, _ = find_peaks(x)
    
    Figure = plt.figure() #全体のグラフを作成
    
    for i in range(60):
        i = i + 1
        ax = Figure.add_subplot(8,8,i) #1つ目のAxを作成
        
        ax.plot(x)
        
    plt.savefig("data/" + "histgram.png")
    plt.show()
        

            

        
    
    
    

