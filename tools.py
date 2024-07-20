"""
Algorithmic and mathematical python tools.
"""

import numpy as np
from scipy import interpolate
import math

# https://stackoverflow.com/questions/16981921/relative-imports-in-python-3
# https://stackoverflow.com/questions/3220755/how-to-find-the-target-files-fullabsolute-path-of-the-symbolic-link-or-soft-l
import sys, os
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__))) # sets directory of python-tools as first python path
from bind import getHistogram, getHistogramLinear               # import from shared library bind in directory of python-tools

#####################
### MISCELLANEOUS ###
#####################

def mean_sterr(values, remove=False, max=None):
    """
    Returns mean and standard error of values.

    Parameters
    ----------
    values : float array
        Values.
    remove : bool
        Remove inf and -inf as well as nan. (default: False)
        NOTE: A warning will be issued if remove == False and such objects are
              encountered.
        NOTE: This is not guaranteed to work non-1D arrays as the shape may
              change.
    max : float or None
        Remove data which is strictly above max in absolute value.
        (default: None)
        NOTE: max != None will trigger remove = True.

    Returns
    -------
    mean : float or float Numpy array
        Mean of values.
    sterr : float or float Numpy array
        Standard error of values = std(...)/sqrt(len(...)).
        NOTE: This is relevant if all values are independent.
    """

    if max != None: remove = True

    values = np.array(values)
    if remove: values = (
        (lambda _: _[~np.isinf(_)])(    # remove inf
        (lambda __: __[~np.isnan(__)])( # remove nan
        values)))
    if max != None: values = values[np.abs(values) <= max]
    if values.size == 0: return None, None

    return values.mean(axis=0), values.std(axis=0)/np.sqrt(values.shape[0])

def wo_mean(arr, axis=-2):
    """
    Returns deviation of values in array with respect to mean on the `axis'-th
    axis if there are more than one value in this dimension.

    Parameters
    ----------
    arr : float array like
        Array of values.
    axis : int
        Axis on which to compute the mean. (default: -2)

    Returns
    -------
    dev_arr : (arr.shape) float numpy array
        Deviations from mean of array.
    """

    arr = np.array(arr)
    if arr.shape[axis] == 1: return arr

    return arr - arr.mean(axis=axis, keepdims=True)

def corr_scale(s, c, threshold=np.exp(-1)):
    """
    Returns scale of correlation function as the extrapolated x-value at which
    its y-value crosses a given threshold.

    Parameters
    ----------
    s : (*,) float array-like
        x-values fo the correlation function.
    c : (*,) float array-like
        y-values fo the correlation function.
    threshold : float
        Threshold which defines the scale of the correlation.
        (default: exp(-1))

    Returns
    -------
    ss : float or numpy.nan
        Scale of the correlation.
        NOTE: ss = numpy.nan when the correlation function does not cross the
              threshold.
    """

    return interpolate.interp1d(c, s, bounds_error=False, fill_value=np.nan)(
        threshold)

class Counter:
    """
    Return integers in order at each call.
    """

    def __init__(self, initial=0):
        """
        Parameters
        ----------
        initial : int
            Starting point of the counter. (default: 0)
        """

        assert type(initial) == int
        self.counter = initial - 1

    def __call__(self):

        self.counter += 1
        return self.counter

class MultiIntKeyDict(object):
    """
    Dictionary-like object in which multiple integer keys are associated to the
    same value. (see https://stackoverflow.com/questions/11105115/)
    """

    def __init__(self, **kwargs):
        """
        Initialises with (possibly empty) dictionary.
        """

        self._keys = {} # secondary dictionary which associates multiple keys to an unique key of self._data
        self._data = {} # values with unique identification

        for k, v in kwargs.items():
            self[k] = v

    def __getitem__(self, key):

        return self._data[self._keys[key]]

    def __setitem__(self, k, v):

        try:

            self._data[self._keys[k]] = v

        except KeyError:

            # check that key in an integer or a tuple of integers
            try:
                assert isinstance(k, int)
                keys = (k,)
            except AssertionError:
                assert isinstance(k, tuple)
                keys = k
                for key in keys:
                    assert isinstance(key, int)

            # unique indices for new value
            try:
                uniqueIndex = max(self._data) + 1
            except ValueError:
                uniqueIndex = 0

            # associate all keys to value
            self._data[uniqueIndex] = v
            for key in keys:
                self._keys[key] = uniqueIndex

    def __delitem__(self, key):

        uniqueIndex = self._keys[key]
        del self._keys[key]
        if not(uniqueIndex in self._keys.values()):
            del self._data[uniqueIndex]

    def __contains__(self, key):

        return self._keys.__contains__(key)

    def __iter__(self):

        return self._keys.__iter__()

    def __next__(self):

        return self._keys.__next__()

    def values(self):

        return self._data.values()

    def add_key(self, existingKey, *newKeys):
        """
        Associates a new key to an existing key.

        Parameters
        ----------
        existingKey : *
            Key already associated to a value.
        newKeys : *
            New keys to associate to the same value.
        """

        uniqueIndex = self._keys[existingKey]
        for newKey in newKeys:
            self._keys[newKey] = uniqueIndex

def progressbar(p, width=40):
    """
    Display progress bar. (https://stackoverflow.com/a/3160819/7385044)

    Parameters
    ----------
    p : float
        Status of the bar (between 0 and 1).
        NOTE: if p == 1 then the progress bar is ended.
    width : int
        Width of the progress bar in number of characters. (default: 40)
    """

    out = "[%s] (%2i%%) " % (
        "="*math.ceil(p*width)
            + " "*(width - math.ceil(p*width)),
        math.ceil(100*p))
    sys.stderr.write(out)
    sys.stderr.flush()
    sys.stderr.write("\b"*len(out))
    if p == 1: sys.stderr.write("\n")   # end the progress bar

#####################
### DISTRIBUTIONS ###
#####################

class Distribution:
    """
    Analyse distribution from array of values.
    """

    def __init__(self, valuesArray):
        """
        Define array of values.

        Parameters
        ----------
        valuesArray : float array-like
            Array of values.
        """

        self.valuesArray = np.array(valuesArray).flatten()

        self.min = self.valuesArray.min()
        self.max = self.valuesArray.max()

        self.mean = self.valuesArray.mean()
        self.std = self.valuesArray.std()

    def hist(self, nBins, vmin=None, vmax=None, log=False,
        rescaled_to_max=False, occupation=False):
        """
        Returns histogram of array of values.

        Parameters
        ----------
        nBins : int
            Number of bins of the histogram.
        vmin : float
            Minimum value of the bins. (default: None)
            NOTE: if vmin==None, then minimum of array is taken.
        vmax : float
            Maximum value of the bins. (default: None)
            NOTE: if vmax==None, then maximum of array is taken.
        log : bool
            Consider the log of the occupancy of the bins. (default: False)
        rescaled_to_max : bool
            Rescale occupancy of the bins by its maximum over bins.
            (default: False)
        occupation : bool
            Return histogram of occupation rather than proportion.
            (default: False)

        Returns
        -------
        bins : float numpy array
            Values of the bins.
        hist : float numpy array
            Occupancy of the bins.
        """

        if vmin == None: vmin = self.min
        if vmax == None: vmax = self.max
        histogram = Histogram(nBins, vmin, vmax, log=False)
        histogram.values = self.valuesArray

        bins = histogram.bins
        hist = histogram.get_histogram(occupation=occupation)
        if rescaled_to_max: hist /= hist.max()
        if not(log): return bins, hist
        else: return bins[hist > 0], np.log(hist[hist > 0])

    def gauss(self, *x, cut=None, rescaled_to_max=False):
        """
        Returns values of the Gaussian function corresponding to the mean and
        variance of the array of values.

        Parameters
        ----------
        x : float
            Values at which to evaluate the Gaussian function.
        cut : float or None
            Width in units of the standard deviation of the array of values to
            consider when computing mean and standard deviation.
            (see self._meanStdCut) (default: None)
            NOTE: if cut==None, the width is taken to infinity, i.e. no value is
                  excluded.
        rescaled_to_max : bool
            Rescale function by its computed maximum. (default: False)

        Returns
        -------
        gauss : float numpy array
            Values of the Gaussian function at x.
        """

        mean, std = self._meanStdCut(cut=cut)

        if rescaled_to_max: norm = 1
        else: norm = np.sqrt(2*np.pi*(std**2))

        gauss = lambda y: (
            np.exp(-((y - mean)**2)/(2*(std**2)))
            /norm)

        return np.array(list(map(gauss, x)))

    def _meanStdCut(self, cut=None):
        """
        Returns mean and standard deviation of values of array with values
        farther than `cut' * self.valuesArray.std() if the mean removed.

        Parameters
        ----------
        array : array-like
            Array of values.
        cut : float
            Width in units of self.valuesArray.std() to consider.
            (default: None)
            NOTE: if cut==None, then no value is excluded.

        Returns
        -------
        mean : float
            Mean of the truncated ensemble.
        std : float
            Standard deviation of the truncated ensemble.
        """

        return meanStdCut(self.valuesArray, cut=cut)

class JointDistribution:
    """
    Analyse joint distribution from 2 arrays of values.
    """

    def __init__(self, valuesArray1, valuesArray2):
        """
        Define array of values.

        Parameters
        ----------
        valuesArray1 : float array-like
            First array of values.
        valuesArray2 : float array-like
            Second array of values.
        """

        self.valuesArray1 = np.array(valuesArray1).flatten()
        self.valuesArray2 = np.array(valuesArray2).flatten()

        self.min1 = self.valuesArray1.min()
        self.max1 = self.valuesArray1.max()
        self.min2 = self.valuesArray2.min()
        self.max2 = self.valuesArray2.max()

    def hist(self, nBins, vmin1=None, vmax1=None, vmin2=None, vmax2=None):
        """
        Returns 3D histogram of arrays of values.

        Parameters
        ----------
        nBins : int or 2-uple-like of int
            Number of bins of the histogram in all or each direction.
        vmin1 : float
            Minimum value of the bins for the first array. (default: None)
            NOTE: if vmin1==None, then minimum of array is taken.
        vmax1 : float
            Maximum value of the bins for the first array. (default: None)
            NOTE: if vmax1==None, then maximum of array is taken.
        vmin2 : float
            Minimum value of the bins for the second array. (default: None)
            NOTE: if vmin2==None, then minimum of array is taken.
        vmax2 : float
            Maximum value of the bins for the second array. (default: None)
            NOTE: if vmax2==None, then maximum of array is taken.

        Returns
        -------
        hist : (nBins.prod(), 3) float Numpy array
            Values of the histogram:
                (0) Bin value of the first quantity.
                (1) Bin value of the second quantity.
                (2) Proportion.
        """

        if vmin1 == None: vmin1 = self.min1
        if vmax1 == None: vmax1 = self.max1
        if vmin2 == None: vmin2 = self.min2
        if vmax2 == None: vmax2 = self.max2
        histogram = Histogram3D(nBins, (vmin1, vmin2), (vmax1, vmax2),
            log=False)
        histogram.values = np.array(list(
            zip(self.valuesArray1, self.valuesArray2)))

        return histogram.get_histogram()

##################
### HISTOGRAMS ###
##################

class Histogram:
    """
    Make histogram from lists of float values.
    """

    def __init__(self, Nbins, vmin, vmax, log=False):
        """
        Parameters
        ----------
        Nbins : int
            Number of histogram bins.
        vmin : float
            Minimum included value for histogram bins.
            NOTE: values lesser than vmin will be ignored.
        vmax : float
            Maximum excluded value for histogram bins.
            NOTE: values greater or equal to vmax will be ignored.
        log : bool.
            Logarithmically spaced histogram values. (default: False)
        """

        self.Nbins = int(Nbins)
        self.vmin = vmin
        self.vmax = vmax

        self.log = log
        if log:
            self.bins = np.logspace(np.log10(self.vmin), np.log10(self.vmax),
                self.Nbins, endpoint=False, base=10)    # histogram bins
        else:
            self.bins = np.linspace(self.vmin, self.vmax,
                self.Nbins, endpoint=False)             # histogram bins

        self.reset_values()                 # reset values from which to compute the histogram
        self.hist = np.empty(self.Nbins)    # values of the histogram at bins

    def add_values(self, *values, replace=False):
        """
        Add values from which to compute the histogram.

        Parameters
        ----------
        values : float or float array-like
            Values to add.
        replace : bool
            Replace existing values. (default: False)
        """

        if replace: self.reset_values()
        for value in values: self.values = np.append(self.values, value)

    def reset_values(self):
        """
        Delete values from which to compute the histogram (self.values).
        """

        self.values = np.array([])

    def get_histogram(self, occupation=False):
        """
        Get histogram from values in self.values.

        Parameters
        ----------
        occupation : bool
            Return histogram of occupation rather than proportion.
            (default: False)

        Returns
        -------
        hist : Numpy array
            Values of the histogram at self.bins.
        """

        if self.log:
            self.hist = getHistogram(
                self.values, self.bins.tolist() + [self.vmax])
        else:
            self.hist = getHistogramLinear(
                self.values, self.Nbins, self.vmin, self.vmax)
        self.hist = self.hist.astype(float)

        binned_values = np.sum(self.hist)
        if binned_values == 0: return self.hist # no binned value
        elif not(occupation):
            # self.hist /= self.hist.sum()
            self.hist /= (
                self.hist*np.diff(self.bins.tolist() + [self.vmax])).sum()
            self.hist *= (
                self.values[
                    (self.values >= self.vmin)*(self.values < self.vmax)].size
                /self.values.size)
        return self.hist

class Histogram3D:
    """
    Make 3D histogram from lists of float 2-uples-like.
    """

    def __init__(self, Nbins, vmin, vmax, log=False):
        """
        Parameters
        ----------
        Nbins : int or 2-uple-like of int
            Number of histogram bins in each direction.
        vmin : float or 2-uple like of float
            Minimum included value for histogram bins.
            NOTE: values lesser than vmin will be ignored.
        vmax : float or 2-uple like of float
            Maximum excluded value for histogram bins.
            NOTE: values greater or equal to vmax will be ignored.
        log : bool.
            Logarithmically spaced histogram values. (default: False)
        """

        Nbins = np.array(Nbins, ndmin=1, dtype=int)
        self.Nbins = np.array([Nbins[0], Nbins[-1]])

        vmin, vmax = np.array(vmin, ndmin=1), np.array(vmax, ndmin=1)
        self.vmin = np.array([vmin[0], vmin[-1]])
        self.vmax = np.array([vmax[0], vmax[-1]])

        self.bins = []
        for dim in range(2):
            if log:
                self.bins += [np.logspace(
                    np.log10(self.vmin[dim]), np.log10(self.vmax[dim]),
                    self.Nbins[dim], endpoint=False, base=10)]  # histogram bins
            else:
                self.bins += [np.linspace(
                    self.vmin[dim], self.vmax[dim],
                    self.Nbins[dim], endpoint=False)]           # histogram bins

        self.reset_values()                             # reset values from which to compute the histogram
        self.hist = np.empty((self.Nbins.prod(), 3))    # values of the histogram at bins
        for bin0 in range(self.bins[0].size):
            self.hist[
                bin0*self.bins[1].size:(bin0 + 1)*self.bins[1].size, 0] = (
                self.bins[0][bin0])
            self.hist[
                bin0*self.bins[1].size:(bin0 + 1)*self.bins[1].size, 1] = (
                self.bins[1])

    def add_values(self, *values, replace=False):
        """
        Add values from which to compute the histogram.

        Parameters
        ----------
        values : float or float array-like
            Values to add.
        replace : bool
            Replace existing values. (default: False)
        """

        if replace: self.reset_values()
        for value in values: self.values += [tuple(value)]

    def reset_values(self):
        """
        Delete values from which to compute the histogram (self.values).
        """

        self.values = []

    def get_histogram(self):
        """
        Get histogram from values in self.values.

        Returns
        -------
        hist : (self.Nbins.prod(), 3) float Numpy array
            Values of the histogram:
                (0) Value of first axis bin.
                (1) Value of second axis bin.
                (2) Proportion.
        """

        values_array = np.array(self.values)
        for bin0 in range(self.bins[0].size):
            bin_inf0 = self.bins[0][bin0]
            try: bin_sup0 = self.bins[0][bin0 + 1]
            except IndexError: bin_sup0 = self.vmax[0]
            values = values_array[
                (values_array[:, 0] >= bin_inf0)
                *(values_array[:, 0] < bin_sup0)][:, 1]
            for bin1 in range(self.bins[1].size):
                bin_inf1 = self.bins[1][bin1]
                try: bin_sup1 = self.bins[1][bin1 + 1]
                except IndexError: bin_sup1 = self.vmax[1]
                self.hist[bin0*self.Nbins[1] + bin1, 2] = (
                    np.sum((values >= bin_inf1)*(values < bin_sup1)))
        self.hist = self.hist.astype(float)

        if np.sum(self.hist[:, 2]) > 0: # there are binned values
            self.hist[:, 2] /= np.sum(self.hist[:, 2])
        return self.hist

#################
### ENVELOPES ###
#################

def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Compute high and low envelopes of a signal `s'.

    https://stackoverflow.com/questions/34235530/how-to-get-high-and-low-envelope-of-a-signal

    Parameters
    ----------
    s : (*,) array-like
        Data signal from which to extract high and low envelopes.
    dmin : int
        Minimum size of chunks. Use this if the size of the input signal is too
        big. (default: 1)
    dmax : int
        Maximum size of chunks. Use this if the size of the input signal is too
        big. (default: 1)
    split: bool
        Split the signal in half along its mean, might help to generate the
        envelope in some cases. (default: False)

    Returns
    -------
    lmin : (**,) numpy int array
        Indices of the data point forming the low envelope of the signal.
    lmax : (***,) numpy int array
        Indices of the data point forming the high envelope of the signal.
    """

    # locals min
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s)
        # pre-sorting of locals min based on relative position with respect to s_mid
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid
        lmax = lmax[s[lmax]>s_mid]

    # global max of dmax-chunks of locals max
    lmin = lmin[[i + np.argmin(s[lmin[i:i + dmin]])
        for i in range(0, len(lmin), dmin)]]
    # global min of dmin-chunks of locals min
    lmax = lmax[[i + np.argmax(s[lmax[i:i + dmax]])
        for i in range(0, len(lmax), dmax)]]

    return lmin, lmax

def hl_envelopes(x, y, dmin=1, dmax=1, split=False):
    """
    Compute interpolation for low and high envelopes of `y(x)' (see
    hl_envelopes_idx).

    https://stackoverflow.com/questions/63541222/fill-between-two-lines-lacking-common-x-values

    Parameters
    ----------
    x : (*,) array-like
        x-axis data of signal from which to extract high and low envelopes.
    y : (*,) array-like
        y-axis data of signal from which to extract high and low envelopes.
    dmin : int
        Minimum size of chunks. Use this if the size of the input signal is too
        big. (default: 1)
    dmax : int
        Maximum size of chunks. Use this if the size of the input signal is too
        big. (default: 1)
    split: bool
        Split the signal in half along its mean, might help to generate the
        envelope in some cases. (default: False)

    Returns
    -------
    newx : (**,) numpy float array
        x-axis data of interpolated signal.
    ymin : (**,) numpy float array
        y-axis data of low envelope of interpolated signal.
    ymax : (**,) numpy float array
        y-axis data of high envelope of interpolated signal.
    """

    x, y = np.array(x), np.array(y)
    assert x.ndim == y.ndim == 1
    assert x.size == y.size

    # low and high envelopes
    lmin, lmax = hl_envelopes_idx(y)
    xmin, ymin = x[lmin], y[lmin]
    xmax, ymax = x[lmax], y[lmax]

    # interpolation
    newx = np.sort(np.concatenate([xmin, xmax]))
    ymin = np.interp(newx, xmin, ymin)
    ymax = np.interp(newx, xmax, ymax)

    return newx, ymin, ymax

