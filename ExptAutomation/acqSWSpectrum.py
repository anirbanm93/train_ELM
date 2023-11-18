"""
Description:
The program
(1) measures spin-wave transmission characteristics.
(2) measures eigen modes of spin-wave ring resonator.
(3) measures the self-generated spin-wave spectrum from the ring oscillator.
(4) measures the output spectrum of the spin-wave ring oscillator driven by microwave generator.

Author:RSA CONTROL: Morgan Allison(tektronics)
       RF-GEN CONTROL: Shubham
       NI-DAQ: NI-DAQ blogs
       Modifications: Anirban Mukhopadhyay

Affiliations: Prof. Anil Prabhakar's Magnonics Lab
"""
from ctypes import *
from RSA_API import *
from RSA_API_funclib import create_frequency_array, config_spectrum, acquire_spectrum, search_connect

import numpy as np
import os
import time
import serial
import nidaqmx
import sys
import shutil

os.add_dll_directory("C:\\Tektronix\\RSA_API\\lib\\x64")
rsa = cdll.LoadLibrary("RSA_API.dll")


class AcqSWSpectrum:

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

    # Propagating spin-wave transmission spectroscopy
    def acqspec_swtx(self, savepath: str, savefilename: str = None, N_rep: int = 1):

        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        if not os.path.isdir(savepath + '\\temp'):
            os.makedirs(savepath + '\\temp')

        if savefilename is None:
            savefilename = savepath.split('\\')[-1]

        start = time.time()

        # initialize RF generator
        rfgen = serial.Serial('COM3', 9600, timeout=None, parity=serial.PARITY_NONE, bytesize=serial.EIGHTBITS)
        if rfgen.isOpen():
            rfgen.close()

        data = np.empty((len(self.genPwr), len(self.genFreq), N_rep))

        # search RSA and connect
        search_connect()

        # configure RF generator
        rfgen.open()
        rfgen.write(str.encode('x2'))  # Setting internal reference to 10 MHz
        rfgen.write(str.encode('C1'))  # Select Channel 0/A or 1/B

        # recording spectra
        for i, elem2 in enumerate(self.genPwr):
            rfgen.write(str.encode('E0r0'))
            rfgen.write(str.encode('W' + str(elem2)))  # Power in dBm
            rfgen.write(str.encode('E1r1'))
            print(f'{i}th RF power input: {elem2} dBm.')

            for j, fin in enumerate(self.genFreq):
                rfgen.write(str.encode('f' + str(fin * 1e-06)))  # freq to RF Gen in MHz
                rfgen.flushInput()
                print(f'{j}th drive frequency point: {fin} Hz.')
                time.sleep(0.5)  # 500 ms wait

                for k in range(N_rep):
                    specSet = config_spectrum(fin, self.reflevel, self.span, self.rbw,
                                              self.tracepts)  # set desired RSA settings
                    traceData = acquire_spectrum(specSet)

                    peakPower = np.amax(traceData)
                    data[i, j, k] = peakPower

                    # Temporary file saving
                    np.save(savepath + '\\temp\\' + 'temp_' + savefilename + str([i, j, k]), peakPower)

                    print(f'Repetition no.: {k}')
                    time.sleep(0.01)  # 10 ms wait between succesive iterations

        # Saving data
        np.save(savepath + '\\' + savefilename, data)

        #######DISCONNECTING##########
        print('Disconnecting spectrum analyzer...')
        rsa.DEVICE_Stop()
        rsa.DEVICE_Disconnect()

        print('Disconnecting RF generator...')
        rfgen.write(str.encode('E0r0'))
        rfgen.close()

        with open(savepath + '\\' + savefilename + '_exptParams.txt', 'w+') as f:
            f.write(f'SA settings: span = {self.span} Hz, RBW = {self.rbw} Hz, Ref. level = {self.reflevel} dBm, '
                    f'Trace pts = {self.tracepts},\n' +
                    f'RF source settings: Input power = {self.genPwr} dBm, Freq. array = {self.genFreq} Hz,\n' +
                    f'Run time: {time.time() - start} seconds'
                    )
            f.close()

        # Deleting temporary files
        print('Deleting temporary folder...')
        shutil.rmtree(savepath + '\\temp')

    # Experiments on spin-wave active ring oscillator (SWARO)
    def acqspec_swaro(self, savepath: str, savefilename: str = None, N_rep: int = 1):

        if not os.path.isdir(savepath):
            os.makedirs(savepath)
               
        if not os.path.isdir(savepath + '\\temp'):
            os.makedirs(savepath + '\\temp')

        if savefilename is None:
            savefilename = savepath.split('\\')[-1]

        start = time.time()

        data = np.empty((N_rep, self.tracepts, 2))

        #################SEARCH RSA /CONNECT#################
        search_connect()

        for i in range(N_rep):
            #################ACQUIRE/SAVE DATA#################
            specSet = config_spectrum(self.fc, self.reflevel,
                                      self.span, self.rbw, self.tracepts)  # set desired settings esa
            traceData = acquire_spectrum(specSet)
            traceFreq = create_frequency_array(specSet) * 1e-09  # Hz to GHz
            data[i, :, 0] = traceFreq
            data[i, :, 1] = traceData

            # Temporary file saving
            np.save(savepath + '\\temp\\' + 'temp_' + savefilename + str(i), [traceFreq, traceData])

            print(f'Repetition no.: {i}.')
            time.sleep(0.01)  # 10 ms wait between succesive iterations

        # Saving data
        np.save(savepath + '\\' + savefilename, data)

        #######DISCONNECTING##########
        print('Disconnecting spectrum analyzer...')
        rsa.DEVICE_Stop()
        rsa.DEVICE_Disconnect()

        with open(savepath + '\\' + savefilename + '_exptParams.txt', 'w+') as f:
            f.write(f'SA settings: span = {self.span} Hz, RBW = {self.rbw} Hz, Ref. level = {self.reflevel} dBm, '
                    f'Trace pts = {self.tracepts},\n' +
                    f'Run time: {time.time() - start} seconds')
            f.close()

        # Deleting temporary files
        print('Deleting temporary folder...')
        shutil.rmtree(savepath + '\\temp')

    def acqSpec_drivenSWARO(self, savepath: str, savefilename: str = None, N_rep: int = 1, get_sgspec: bool = False):

        if not os.path.isdir(savepath):
            os.makedirs(savepath)
               
        if not os.path.isdir(savepath + '\\temp'):
            os.makedirs(savepath + '\\temp')

        if savefilename is None:
            savefilename = savepath.split('\\')[-1]

        start = time.time()

        rfgen = serial.Serial('COM3', 9600, timeout=None, parity=serial.PARITY_NONE, bytesize=serial.EIGHTBITS)
        if rfgen.isOpen():
            rfgen.close()

        data = np.empty((len(self.genPwr), len(self.genFreq), N_rep, self.tracepts, 2))

        #################SEARCH RSA /CONNECT#################
        search_connect()

        #################CONFIG RFGEN#################
        rfgen.open()
        rfgen.write(str.encode('x2'))  # Setting internal reference to 10 MHz
        rfgen.write(str.encode('C1'))  # Select Channel 0/A or 1/B

        if get_sgspec:
            self.acqspec_swaro(savepath=savepath, savefilename=savefilename + '_sgspec', N_rep=N_rep)
            print('self-generation spectrum is obtained; Driving of SWARO will start now')

        # Driven SWARO
        for i, elem2 in enumerate(self.genPwr):
            rfgen.write(str.encode('E0r0'))
            rfgen.write(str.encode('W' + str(elem2)))  # Power in dBm
            rfgen.write(str.encode('E1r1'))
            print(f'{i}th RF power input: {elem2} dBm.')

            for j, fin in enumerate(self.genFreq):
                rfgen.write(str.encode('f' + str(fin * 1e-06)))  # freq to RF Gen in MHz
                rfgen.flushInput()
                print(f'{k}th drive frequency point: {fin} Hz.')
                time.sleep(0.5)  # wait period of 500 ms

                for k in range(N_rep):
                    #################ACQUIRE/SAVE DATA#################
                    specSet = config_spectrum(self.fc, self.reflevel, self.span, self.rbw,
                                              self.tracepts)
                    traceData = acquire_spectrum(specSet)
                    traceFreq = create_frequency_array(specSet) * 1e-09  # Hz to GHz
                    data[i, j, k, :, 0] = traceFreq
                    data[i, j, k, :, 1] = traceData

                    # Temporary file saving
                    np.save(savepath + '\\temp\\' + 'temp_' + savefilename + str([i, j, k]), [traceFreq, traceData])

                    print(f'Repetition no.: {k}.')
                    time.sleep(0.01)  # 10 ms wait between successive iterations

        # Saving data
        np.save(savepath + '\\' + savefilename, data)

        #######DISCONNECTING##########
        print('Disconnecting spectrum analyzer...')
        rsa.DEVICE_Stop()
        rsa.DEVICE_Disconnect()

        print('Disconnecting RF generator...')
        rfgen.write(str.encode('E0r0'))
        rfgen.close()

        with open(savepath + '\\' + savefilename + '_exptParams.txt', 'w+') as f:
            f.write(f'SA settings: span = {self.span} Hz, RBW = {self.rbw} Hz, Ref. level = {self.reflevel} dBm, '
                    f'Trace pts = {self.tracepts},\n' +
                    f'RF source settings: Input power = {self.genPwr} dBm, Freq. array = {self.genFreq} Hz,\n' +
                    f'Run time: {time.time() - start} seconds'
                    )
            f.close()

        # Deleting temporary files
        print('Deleting temporary folder...')
        shutil.rmtree(savepath + '\\temp')
