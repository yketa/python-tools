#ifndef BIND_HPP
#define BIND_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

/*
 *  C++ helper functions
 *
 */

template<class T, class TT> T const _pmod(T const& i, TT const& n)
/*
Positive modulo (remainder) i%n (https://stackoverflow.com/questions/14997165).
*/
    { return std::fmod(std::fmod(i, n) + n, n); }

template<class T> bool _inVec(std::vector<T> const& vec, T const& val)
/*
Is `val' in `vec'?
*/
    { return (std::find(vec.begin(), vec.end(), val) != vec.end()); }

template<class T> long int _indexVec(std::vector<T> const& vec, T const& val) {
/*
Return index of `val' in `vec' or raise assertion error if it is absent.
*/

    typename std::vector<T>::const_iterator
        it = std::find(vec.begin(), vec.end(), val);
    assert(it != vec.end());    // assert value is present
    return it - vec.begin();    // value index
}

std::vector<double> _get2DL(pybind11::array_t<double> const& L) {
/*
Return (2,) box size dimensions from array `L' which may be a double or an
array of double.
*/

    auto l = L.unchecked<>();
    if (l.ndim() > 1) {                             // system size is 2D or higher-D: impossible
        throw std::invalid_argument("L must be a 0- or 1-dimensional.");
    }
    else if (l.size() == 0) {                       // no system size: impossible
        throw std::invalid_argument("System size cannot be empty.");
    }
    else if (l.size() == 1) {                       // system size is the same for both dimensions
        if (l.ndim() == 0) {
            return std::vector<double>({l(), l()});
        }
        else {  // l.ndim() == 1
            return std::vector<double>({l(0), l(0)});
        }
    }
    else {                                          // system size is different for the two dimensions
        return std::vector<double>({l(0), l(1)});
    }
}

template<class T> std::vector<long int> _argSort(
    std::vector<T> const& vec) {
/*
Indices of elements of `vec' in increasing order.
https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
*/

    // initialise original index locations
    std::vector<long int> indices(vec.size());
    std::iota(indices.begin(), indices.end(), 0);

    // sort indices based on comparing values in `vec'
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when `vec' contains elements of equal values
    std::stable_sort(indices.begin(), indices.end(),
        [&vec](long int const& i1, long int const& i2)
            { return vec[i1] < vec[i2]; });

    return indices;
}

template<class T, class TT> std::vector<T> _keySort(
    std::vector<T> const& vec, std::vector<TT> const& keys) {
/*
Sort elements of `vec' in increasing order of elements of keys.
*/

    std::vector<T> sortedVec;
    for (long int i : _argSort<TT>(keys)) sortedVec.push_back(vec[i]);

    return sortedVec;
}

double angle2(double const& x, double const& y) {
    double const angle = acos(x/sqrt(x*x + y*y));
    return _pmod(((y > 0 ? 1 : - 1)*angle)
        + std::numbers::pi, 2*std::numbers::pi)
        - std::numbers::pi;
}

/*
 *  Fourier transforms
 *
 */

pybind11::array_t<double> getAllWaveVectors2D(
    pybind11::array_t<double> const& L,
    double const& qmin, double const& qmax,
    int const& nmax=0) {
/*
Return wave vectors associated to rectangular box of size `L' such that their
norms belong to [`qmin', `qmax'].
Only a single vector of each pair of opposite wave vectors is returned.
*/

    // check system size
    std::vector<double> const systemSize = _get2DL(L);
    double const qx = 2*std::numbers::pi/systemSize[0];
    double const qy = 2*std::numbers::pi/systemSize[1];

    // loop in wave vector space
    std::vector<std::vector<double>> waveVectors(0);
    long long int const xmin = 0;
    long long int const xmax = floor(qmax/qx);
    for (long long int x=xmin; x <= xmax; x++) {
        long long int const ymin =
            std::max(
                0.,
                ceil(sqrt(pow(qmin, 2) - pow(qx*x, 2))/qy));
        long long int const ymax =
            std::min(
                floor(qmax/qy),
                floor(sqrt(pow(qmax, 2) - pow(qx*x, 2))/qy));
        for (long long int y=ymin; y <= ymax; y++) {
            double const qq = sqrt(pow(qx*x, 2) + pow(qy*y, 2));
            if (qq < qmin || qq > qmax) { continue; }   // wave vector norm not within interval
            waveVectors.push_back({qx*x, qy*y});
            if (x != 0 && y != 0) {
                // if x == 0 then (qx x, qy y) and (-qx x, qy y) are identical
                // if y == 0 then (qx x, qy y) and (-qx x, qy y) are opposite
                waveVectors.push_back({-qx*x, qy*y});
            }
        }
    }

    // create and return array
    long int const total_size = waveVectors.size();
    if (total_size == 0) { return pybind11::array_t<double>(); }
    long int const stride =
        (long int) [&total_size, &nmax]() {
            if (nmax <= 0) return 1.;
            else return std::ceil((double) total_size/(double) nmax);
        }();
    long int const size = (long int) std::floor(total_size/stride);
    pybind11::array_t<double> arr(std::vector<ptrdiff_t>{size, 2});
    auto a = arr.mutable_unchecked<2>();
    for (long int k=0; k < size; k++) {
        for (int dim=0; dim < 2; dim++) {
            a(k, dim) = waveVectors[k*stride][dim];
        }
    }
    return arr;
}

pybind11::array_t<double> getWaveVectors2D(
    pybind11::array_t<double> const& L,
    double const& q, double const& dq=0.1)
    { return getAllWaveVectors2D(L, q - dq/2, q + dq/2); }
/*
Return wave vectors associated to rectangular box of size `L' such that their
norms belong to [`q' - `dq'/2, `q' + `dq'/2].
Only a single vector of each pair of opposite wave vectors is returned.
*/

#endif

