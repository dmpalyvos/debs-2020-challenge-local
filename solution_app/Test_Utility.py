import numpy as np
import pandas as pd
import sklearn.preprocessing as skpre
import sklearn.model_selection as skmose
import random
import os
import platform
from pathlib import Path
import matplotlib
if platform.system() == "Linux": #for matplotlib on Linux
    matplotlib.use('Agg')

import abc

import pdb

import glob

from datetime import timedelta

from io import StringIO
from datetime import datetime

from copy import deepcopy

#-------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------#

class Electrical_Metrics:
    def __init__(self):
        pass

    def active_power(self,instant_voltage, instant_current,period_length):
        """
        Active or Real power is the average of instantaneous power.
        P = Sum ( i[n] * v[n] ) / N )
        First we calculate the instantaneous power by multiplying the instantaneous
        voltage measurement by the instantaneous current measurement. We sum the
        instantaneous power measurement over a given number of samples and divide by
        that number of samples.

        Args:
            instant_voltage: numpy array
            instant_current: numpy array

        Returns:
            active power: numpy array
        """
        instant_current = np.array(instant_current).flatten()
        instant_voltage = np.array(instant_voltage).flatten()

        if len(instant_current) == len(instant_voltage):
            instant_power = instant_voltage * instant_current
            period_length = int(period_length)

            active_power = []
            for i in range(0, len(instant_power), period_length):
                if i + period_length <= len(instant_power):
                    signal_one_period = instant_power[i:int(i + period_length)]
                    active_power_one_period = np.mean(signal_one_period )
                    active_power.append(active_power_one_period)
            active_power = np.array(active_power)
            return active_power

        else:
            raise ValueError("Signals need to have the same length")


    def apparent_power(self, instant_voltage,instant_current,period_length):
        """
        S = Vrms * Irms
        Args:
            instant_voltage: numpy array
            instant_current: numpy array
            period_length: numpy array

        Returns:
            apparent power: numpy array
        """
        if len(instant_current) == len(instant_voltage):

            rms_voltage = self.compute_single_rms(instant_voltage,period_length)
            rms_current = self.compute_single_rms(instant_current,period_length)
            apparent_power = rms_voltage * rms_current
            return apparent_power

        else:
            raise ValueError("Signals need to have the same length")


    def reactive_power(self,apparent_power,active_power):
        """

        Q = sqrt(S^2 - P^2)
        Args:
            apparent_power: numpy array
            active_power: numpy array

        Returns:
            reactive power: numpy array

        """
        if len(apparent_power) == len(active_power):
            reactive_power = np.sqrt((apparent_power * apparent_power) - (active_power * active_power))
            return reactive_power
        else:
            raise ValueError("Signals need to have the same length")

    def compute_power_factor(self,apparent_power,active_power):
        """
        PF = P / S
        Args:
            apparent_power: numpy array
            active_power: numpy array

        Returns:
            power factor: single integer
        """

        power_factor = active_power / apparent_power
        return power_factor

    def compute_voltage_current_rms(self, voltage, current, period_length):
        """

        Args:
            voltage: numpy array
            current: numpy array
            period_length: integer

        Returns:
            voltage_rms: numpy array
            current_rms: numpy array
        """
        period_length = int(period_length)
        voltage_rms = self.compute_single_rms(voltage, period_length)
        current_rms = self.compute_single_rms(current, period_length)
        return voltage_rms, current_rms

    def compute_single_rms(self,signal,period_length):
        """

        Args:
            signal: numpy array
            period_length: in samples: can be the net frequency or a multiple of it

        Returns:
            rms_values
        """
        rms_values = []
        period_length = int(period_length)
        for i in range(0, len(signal), period_length):
            if i + period_length <= len(signal):
                signal_one_period = signal[i:int(i + period_length)]
                rms_one_period = np.sqrt(np.mean(np.square(signal_one_period))) #rms
                rms_values.append(rms_one_period)
        return np.array(rms_values)

