from operator import itemgetter
import sklearn
from sklearn.decomposition import FastICA, PCA
import numpy as np
import scipy.signal

class Processor():
    
    def _prevalent_freq(self, data, framerate):
        """
        Return the most prevalent frequency using power spectrum
        """
        data = (data-np.mean(data))/np.std(data)
        transform = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(len(data), 1.0/framerate)
        freqs = 60*freqs
        band_pass = np.where((freqs < 40) | (freqs > 240) )[0]
        transform[band_pass] = 0
        transform = np.abs(transform)**2
        sos = scipy.signal.butter(3, 0.13, output='sos')
        filtered = scipy.signal.sosfilt(sos, transform)
        id = np.argmax(filtered)
        heart_rate = freqs[id]
        return heart_rate, np.max(filtered)

class ICAProcessor(Processor):

    def get_hr(self, values, framerate):
        ica = FastICA(n_components=3, max_iter=4000)
        signals = ica.fit_transform(values)
        freqs = [self._prevalent_freq(signals[:,i], framerate) for i in range(3)]
        return freqs
        # return self._select_maximum_power_frequency([self._prevalent_freq(signals[:,i], framerate) for i in range(3)])

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