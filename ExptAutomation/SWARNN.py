"""
Description: Physical Neural Network (PhNN) implemented in Spin-Wave Active Ring

Author:RSA CONTROL: Morgan Allison(tektronics)
       RF-GEN CONTROL: Shubham
       NI-DAQ: NI-DAQ blogs
       Modifications and Moku automation: Anirban Mukhopadhyay

Affiliations: Prof. Anil Prabhakar's Magnonics Lab
"""
import numpy as np
import os
import shutil
import time
import serial
import sys
from RSA_API import *
from RSA_API_funclib import config_spectrum, acquire_spectrum, search_connect
from moku.instruments import ArbitraryWaveformGenerator

os.add_dll_directory("C:\\Tektronix\\RSA_API\\lib\\x64")
rsa = cdll.LoadLibrary("RSA_API.dll")


def gen_sumsinesigs(freq_list: list | np.ndarray, weight: list | np.ndarray):
    """
    param freq_list: a list of user given frequencies
    return: a sum of sinusoidal signals with user given frequencies
    """
    if isinstance(freq_list, list):
        freq_list = np.array(freq_list)
        
    # Find the maximum frequency
    fmax = max(freq_list)

    # Sampling period. Consider 10 samples per period for sinusoidal signals
    Ts = 1 / (10 * fmax)

    # Greatest common divisor of user given frequencies
    freq_list = freq_list.astype(int)
    print(f'IF frequencies: {freq_list} Hz')

    f0 = np.gcd.reduce(freq_list)
    T0 = 1 / f0

    print(f'We are sampling the signal with a period of {T0 * 1e06} mus at a rate {Ts * 1e09} ns.'
          f'\nNumber of sample points are {int(T0 / Ts) + 1}.')

    # Time array. Maximum data length is 65535 samples
    if int(T0 / Ts) + 1 > 65535:
        time_arr = np.linspace(0, T0, 65535)
    else:
        time_arr = np.arange(0, T0 + Ts, Ts)

    # Sum of sinusoidal signals
    print(f'The weight values are: {weight}')

    sig = np.zeros_like(time_arr)
    for w, f in zip(weight, freq_list):
        sig += w * np.sin(2 * np.pi * f * time_arr)

    # Normalize the array between -1 and 1
    sig = sig / np.abs(sig).max()

    return sig, f0


def dbm2vpp(pwrdbm: list | np.ndarray):
    if isinstance(pwrdbm, list):
        pwrdbm = np.array(pwrdbm)
    return 2 * 10 ** (pwrdbm / 20 - 0.5)


class SWARNN:

    # define the experimental parameters
    def __init__(self, **kwargs):
		
        # RF generator settings: frequency in Hz and power in dBm
        if kwargs.get('RFGpwr') is None and kwargs.get('RFGfreq') is None:
            print('No RF input is required for self-generation phenomenon in SWARO.')
        else:
            # To avoid floating point arithmatic error
            self.genFreq = np.round(kwargs['RFGfreq'])
            self.genPwr = kwargs['RFGpwr'] if max(kwargs['RFGpwr']) < 20 else sys.exit('WARNING: Greater than maximum '
                                                                                       'power limit. Terminating...')

        # RSA settings: frequency in Hz and reference in dBm
        if kwargs.get('SAcenterfreq') is None:
            print("spectrum analyzer center frequency = input RF frequency.")
        else:
            # To avoid floating point arithmatic error
            self.fc = np.round(kwargs['SAcenterfreq'])

        self.reflevel = kwargs['SAref']
        self.rbw = kwargs['SArbw']
        self.tracepts = kwargs['SAtracepts']
        self.span = kwargs['SAspan']

    def elm_sinedrive(self, savepath: str, savefilename: str=None):
        """
        param savepath: path where data with the power spectra will be saved.
        return: drives the SWARO and records the spectra.
        """

        if not os.path.isdir(savepath):
            os.makedirs(savepath)
            os.makedirs(savepath + '\\temp')

        if savefilename is None:
            savefilename = savepath.split('\\')[-1]

        start = time.time()

        rfgen = serial.Serial('COM3', 9600, timeout=None, parity=serial.PARITY_NONE, bytesize=serial.EIGHTBITS)
        if rfgen.isOpen():
            rfgen.close()

        data = np.zeros((len(self.genFreq), self.tracepts))

        #################SEARCH RSA /CONNECT#################
        search_connect()

        #################CONFIG RFGEN#################
        rfgen.open()
        rfgen.write(str.encode('x2'))  # Setting internal reference to 10 MHz
        rfgen.write(str.encode('C1'))  # Select Channel 0/A or 1/B

        for k, (pin, fin) in enumerate(zip(self.genPwr, self.genFreq)):
            rfgen.write(str.encode('E0r0'))
            rfgen.write(str.encode('W' + str(pin)))  # Power in dBm
            rfgen.write(str.encode('E1r1'))
            print('RF power input: %0.2f dBm' % pin)
            
            rfgen.write(str.encode('f' + str(fin * 1e-06)))  # freq to RF Gen in MHz
            rfgen.flushInput()
            print('drive frequency point: %0.2f Hz' % fin)
            
            time.sleep(0.5)  # wait period of 200 ms

            #################ACQUIRE/SAVE DATA#################
            specSet = config_spectrum(self.fc, self.reflevel, self.span, self.rbw,
                                      self.tracepts)
            traceData = acquire_spectrum(specSet)

            # Temporary file saving
            np.save(savepath + '\\temp\\' + 'temp_' + savefilename + str(k), traceData)
            data[k, :] = traceData

        # Saving data
        np.save(savepath + '\\' + savefilename, data)

        # Disconnecting
        print('Disconnecting spectrum analyzer...')
        rsa.DEVICE_Stop()
        rsa.DEVICE_Disconnect()

        print('Disconnecting RF generator...')
        rfgen.write(str.encode('E0r0'))
        rfgen.close()

        with open(savepath + '\\' + savefilename + '_exptParams.txt', 'w+') as f:
            f.write(f'SA settings: span = {self.span} Hz, RBW = {self.rbw} Hz, Ref. level = {self.reflevel} dBm, '
                    f'Trace pts = {self.tracepts},\n'+
                    f'RF source settings: Input power = {self.genPwr} dBm, Freq. array = {self.genFreq} Hz,\n'
                    + f'Run time: {time.time() - start} seconds.')
            f.close()

        # Deleting temporary files
        shutil.rmtree(savepath + '\\temp')

    def elm_mixeddrive(self, savepath: str, f_IF: np.ndarray, Vpp_IF: np.ndarray,
                       weight: list | np.ndarray = None, savefilename: str=None):
		
        """
        param savepath: path where data with the power spectra will be saved.
        param f_IF: Each row has a list of frequencies for a specific input. Shape: (Num of instances, Num of features)
        param Vpp_IF: 1D array of IF power.
        param weight: weight of each sinusoidal input. Shape: (Num of instances, Num of features)
        return: drives the SWARO and records the spectra.
        """

        start = time.time()

        if max(Vpp_IF) < 2:

            if not os.path.isdir(savepath):
                os.makedirs(savepath)
            if not os.path.isdir(savepath + '\\temp'):
                os.makedirs(savepath + '\\temp')

            if savefilename is None:
                savefilename = savepath.split('\\')[-1]

            rfgen = serial.Serial('COM3', 9600, timeout=None, parity=serial.PARITY_NONE, bytesize=serial.EIGHTBITS)
            if rfgen.isOpen():
                rfgen.close()

            data = np.zeros((len(f_IF), self.tracepts))
            
            # To avoid floating point arithmatic error
            # Example:
            # (66.1 * 1e06) = 66099999.99999999
            # np.round(66.1 * 1e06) = 66100000
            f_IF = np.round(f_IF)

            # SEARCH RSA /CONNECT
            search_connect()

            # config LO port of the mixer
            rfgen.open()
            rfgen.write(str.encode('x2'))  # Setting internal reference to 10 MHz
            rfgen.write(str.encode('C1'))  # Select Channel 0/A or 1/B

            # LO input
            rfgen.write(str.encode('E0r0'))
            rfgen.write(str.encode('W' + str(self.genPwr[0])))  # 10 dBm power to LO port for level-10 mixer
            rfgen.write(str.encode('E1r1'))
            print('LO power input: %0.2f dBm' % self.genPwr[0])

            rfgen.write(str.encode('f' + str(self.genFreq[0] * 1e-06)))  # freq to RF Gen in MHz
            rfgen.flushInput()
            print('LO frequency: {} Hz'.format(self.genFreq[0]))

            # config IF port of the mixer
            # Connect to your Moku by its ip address ArbitraryWaveformGenerator('192.168.###.###')
            # or by its serial ArbitraryWaveformGenerator(serial=123)
            awg = ArbitraryWaveformGenerator('[fe80::ee24:b8ff:fe16:445f%252518]', force_connect=True)

            # channel 1 load is 50 Ohm
            awg.set_output_load(1, "50Ohm")

            # Out channel-2 is disabled
            awg.enable_output(2, False)

            if weight is None:
                weight = np.ones_like(f_IF)

            for k, (f, w, vpp) in enumerate(zip(f_IF, weight, Vpp_IF)):
                print('Loop number:', k)

                # IF frequency
                sig, f0 = gen_sumsinesigs(f, w)

                # IF power converted into peak to peak voltage
                # Amplitude ranges from +/- 1 V into 50 Ohm
                # Peak to peak amplitude, Maximum is 2 Vpp
                print('Peak to peak voltage: {} V'.format(vpp))

                # We have configurable on-device linear interpolation between LUT points.
                awg.generate_waveform(channel=1, sample_rate='Auto',
                                      lut_data=list(sig), frequency=float(f0),
                                      amplitude=vpp, strict=False)

                time.sleep(0.5)  # wait period of 500 ms

                specSet = config_spectrum(self.fc, self.reflevel, self.span, self.rbw, self.tracepts)
                traceData = acquire_spectrum(specSet)

                # Temporary file saving
                np.save(savepath + '\\temp\\' + 'temp_' + savefilename + str(k), traceData)
                data[k, :] = traceData

            # Saving data
            np.save(savepath + '\\' + savefilename, data)

            # Disconnecting
            print('Disconnecting spectrum analyzer...')
            rsa.DEVICE_Stop()
            rsa.DEVICE_Disconnect()

            print('Disconnecting RF generator...')
            rfgen.write(str.encode('E0r0'))
            rfgen.close()

            print('Disconnecting Moku:Lab AWG...')
            awg.enable_output(1, False)
            awg.relinquish_ownership()

            with open(savepath + '\\' + savefilename + '_exptParams.txt', 'w+') as f:
                f.write(f'SA settings: span = {self.span} Hz, RBW = {self.rbw} Hz, Ref. level = {self.reflevel} dBm, '
                        f'Trace pts = {self.tracepts},\n'+
                        f'RF source settings: Input power = {self.genPwr} dBm, Freq. array = {self.genFreq} Hz,\n'
                        + f'Run time: {time.time() - start} seconds.')
                f.close()

            # Deleting temporary files
            print('Deleting temporary folder...')
            shutil.rmtree(savepath + '\\temp')

        else:
            sys.exit('WARNING: Greater than maximum peak to peak voltage limit of Moku:Lab AWG. Terminating...')
