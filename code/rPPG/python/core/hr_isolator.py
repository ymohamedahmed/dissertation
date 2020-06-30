from operator import itemgetter
import matplotlib 
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import FastICA, PCA
import numpy as np
import scipy.signal
import heartpy as hp

def hr_from_array(freqs):
    powers = [freqs[0][1], freqs[1][1], freqs[2][1]]
    hrs = [freqs[0][0], freqs[1][0], freqs[2][0]]
    i = np.argsort(powers)
    hr_max_power = hrs[i[-1]]
    mean_of_non_max = 1/2 * (np.sum(hrs)- hr_max_power)
    if abs(hr_max_power - mean_of_non_max)/hr_max_power > 0.3 and hr_max_power == max(hrs):
        return hrs[i[-2]]
    else:
        return hr_max_power

class Processor():
    
    def _prevalent_freq(self, data, framerate):
        """
        Return the most prevalent frequency using power spectrum
        """
        if not(np.std(data) == 0):
            data = (data-np.mean(data))/np.std(data)
        transform = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(len(data), 1.0/framerate)   
        freqs = 60*freqs
        band_pass = np.where((freqs < 40) | (freqs > 240) )[0]
        transform[band_pass] = 0
        transform = np.abs(transform)**2
        sos = scipy.signal.butter(3, 0.2, output='sos')
        transform = scipy.signal.sosfilt(sos, transform)
        powers = np.argsort(-1*transform)
        hr, power = self._respiration_rejection([freqs[powers[0]], freqs[powers[1]]],[transform[powers[0]], transform[powers[1]]])
        return hr, power

    def _respiration_rejection(self, peaks, powers):
        """
        Take the two largest peaks and decide which to pick
        """
        peak_hr, lower_peak_hr = peaks[0], peaks[1]
        largest_power, second_power = powers[0], powers[1]
        correct_hr, correct_power = peak_hr, largest_power
        if peak_hr < lower_peak_hr and lower_peak_hr > 90 and peak_hr < 90 and second_power > 0.7*correct_power:
            print(f"Respiration rejection ({peak_hr},{largest_power}) ({lower_peak_hr},{second_power})")
            correct_hr = lower_peak_hr
            correct_power = second_power

        return correct_hr, correct_power


class ICAProcessor(Processor):

    def get_hr(self, values, framerate):
        ica = FastICA(n_components=3, max_iter=40000)
        values = self.remove_nans_and_infs(values)
        signals = ica.fit_transform(values)
        freqs = [self._prevalent_freq(signals[:,i], framerate) for i in range(3)]
        return freqs
    
    def remove_nans_and_infs(self, values):
        length,_ = values.shape
        xp = np.arange(length)
        for i in range(3):
            nan_indices = xp[np.isnan(values[:,i]) | (np.isinf(values[:,i]))]
            nans = np.isnan(values[:,i])
            nan_indices = xp[nans]
            values[nan_indices,i] = np.interp(nan_indices, xp[~nans], values[~nans,i]) 
        return values

    def _select_maximum_power_frequency(self, rates):
        return max(rates, key=itemgetter(1))[0]
    
    def __str__(self):
        return self.__class__.__name__

class PCAProcessor(Processor):
    
    def get_hr(self, values, framerate):
        pca = PCA()
        pca_result = pca.fit_transform(values)
        freqs = [self._prevalent_freq(pca_result[:,i], framerate) for i in range(3)]
        return freqs

    def __str__(self):
        return self.__class__.__name__

class PrimitiveProcessor(Processor):

    def get_hr(self, values, framerate):
        return [self._prevalent_freq(values[:,i], framerate) for i in range(3)]

    def __str__(self):
        return self.__class__.__name__