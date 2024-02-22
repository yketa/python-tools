/*
Tools library written in C++ and exposed to Python with pybind11.
*/

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <complex>
#include <vector>

#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace std::complex_literals;  // using comlex numbers

/*
 *  MISCELLANEOUS
 *
 */

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

/*
 *  Fourier transforms
 *
 */

pybind11::array_t<double> getWaveVectors2D(
    pybind11::array_t<double> const& L,
    double const& q, double const& dq=0.1) {
/*
Return wave vectors associated to rectangular box of size `L' such that their
norms belong to [`q' - `dq'/2, `q' + `dq'/2].
Only a single vector of each pair of opposite wave vectors is returned.
*/

    // check system size
    std::vector<double> const systemSize = _get2DL(L);
    double const qx = 2*std::numbers::pi/systemSize[0];
    double const qy = 2*std::numbers::pi/systemSize[1];

    // loop in wave vector space
    std::vector<std::vector<double>> waveVectors(0);
    long long int const xmin = 0;
    long long int const xmax = floor((q + dq/2)/qx);
    for (long long int x=xmin; x <= xmax; x++) {
        long long int const ymin =
            std::max(
                0.,
                ceil(sqrt(pow(q - dq/2, 2) - pow(qx*x, 2))/qy));
        long long int const ymax =
            std::min(
                floor((q + dq/2)/qy),
                floor(sqrt(pow(q + dq/2, 2) - pow(qx*x, 2))/qy));
        for (long long int y=ymin; y <= ymax; y++) {
            double const qq = sqrt(pow(qx*x, 2) + pow(qy*y, 2));
            if (abs(qq - q) > dq/2) { continue; }   // wave vector norm not within interval
            waveVectors.push_back({qx*x, qy*y});
            if (x != 0 && y != 0) {
                // if x == 0 then (qx x, qy y) and (-qx x, qy y) are identical
                // if y == 0 then (qx x, qy y) and (-qx x, qy y) are opposite
                waveVectors.push_back({-qx*x, qy*y});
            }
        }
    }

    // create and return array
    if (waveVectors.size() == 0) { return pybind11::array_t<double>(); }
    pybind11::array_t<double>
        arr(std::vector<ptrdiff_t>{(long long int) waveVectors.size(), 2});
    auto a = arr.mutable_unchecked<2>();
    for (long long int l=0; l < (long long int) waveVectors.size(); l++) {
        for (int dim=0; dim < 2; dim++) {
            a(l, dim) = waveVectors[l][dim];
        }
    }
    return arr;
}

pybind11::array_t<std::complex<double>> getFT2D(
    pybind11::array_t<double> const& positions,
    pybind11::array_t<double> const& L,
    pybind11::array_t<std::complex<double>> const& values,
    double const& q, double const& dq=0.1) {
/*
Return 2D Fourier transform of `values' associated to 2D `positions' at 2D wave
vectors `q'.
*/

    // wave vector norms
    pybind11::array_t<double> qARR = getWaveVectors2D(L, q, dq);
    if (qARR.request().size == 0)
        { return pybind11::array_t<std::complex<double>>(0); }
    auto _q = qARR.unchecked<2>();      // direct access to wave vectors
    long int const n = _q.shape(0);

    // check positions and values arrays
    auto r = positions.unchecked<2>();  // direct access to positions
    assert(r.ndim() == 2);
    assert(r.shape(1) == 2);
    long int const N = r.shape(0);
    auto v = values.unchecked<>();      // direct access to first values
    if (v.shape(0) != N) {
        throw std::invalid_argument("Positions and values must have = sizes.");
    }
    if (v.ndim() > 1) {
        throw std::invalid_argument("Values must be 1-dimensional.");
    }

    // compute Fourier transform
    pybind11::array_t<std::complex<double>> ft({n});
    auto FT = ft.mutable_unchecked<1>();
    for (long int l=0; l < n; l++) {        // loop over wave vectors
        FT(l) = 0;                                                              // initialise
        for (long int i=0; i < N; i++) {    // loop over particles
            FT(l) += v(i)*std::exp(-1i*(_q(l, 0)*r(i, 0) + _q(l, 1)*r(i, 1)));  // Fourier term
        }
    }

    return ft;
}

/*
 *  Correlations
 *
 */

pybind11::array_t<std::complex<double>> getRadialCorrelations2D(
    pybind11::array_t<double> const& positions,
    pybind11::array_t<double> const& L,
    pybind11::array_t<std::complex<double>> const& values1,
    pybind11::array_t<std::complex<double>> const& values2_,
    int const& nBins,
    bool const& periodic_boundary_conditions=true,
    double const& rmin=0, double const& rmax_=0,
    bool const& rescale_pair_distribution=false) {
/*
Return 2D radial correlations of `values1' and `values2' (1D or 2D).
*/

    // check system size
    std::vector<double> const systemSize = _get2DL(L);

    // system area (for normalisation)
    double const area = [&systemSize]() {
        if (systemSize[0]*systemSize[1] == 0) {
            return 1.;
        }
        else {
            return systemSize[0]*systemSize[1];
        }
    }();

    // check maximum radius
    double const rmax = rmax_ == 0 ?                                // is rmax_ == 0?
        *std::min_element(systemSize.begin(), systemSize.end())/2   // then rmax = min(L)/2
        : rmax_;                                                    // otherwise rmax = rmax_
    if (rmin == rmax) {
        throw std::invalid_argument("Min- and maximum radii cannot be equal.");
    }

    // initialise returned histograms and computation quantities
    pybind11::array_t<std::complex<double>> correlations({nBins, 2});
    auto c = correlations.mutable_unchecked<2>();               // direct access to correlation histogram
    for (int bin=0; bin < c.shape(0); bin++) {
        c(bin, 0) = rmin + (bin + 1./2.)*(rmax - rmin)/nBins;   // centre of correlation histogram bins
        c(bin, 1) = 0;                                          // height of correlation histogram bins
    }
    std::vector<int> occupancy(nBins, 0);                       // occupancy of histogram bins (for normlaisation)
    int nPairs = 0;                                             // number of unique pairs of points (for normalisation)

    // check positions and values arrays
    pybind11::array_t<std::complex<double>> const values2 =
        [&values1, &values2_]() {
            if (values2_.unchecked<>().size() == 0) {   // is second values an empty array?
                return values1;                         // then copy it from first values
            }
            else {
                return values2_;                        // otherwise use input second values
            }
    }();
    auto r = positions.unchecked<2>();                  // direct access to positions
    auto v1 = values1.unchecked<>();                    // direct access to first values
    auto v2 = values2.unchecked<>();                    // direct access to second values
    if (v1.shape(0) != r.shape(0) ||
        v1.shape(0) != v2.shape(0) || v1.size() != v2.size()) {
        throw std::invalid_argument("Values' sizes are not consistent.");
    }
    if (v1.ndim() > 2 || v2.ndim() > 2) {
        throw std::invalid_argument("Values must be 1- or 2-dimensional.");
    }
    bool const values_is_1D = (v1.ndim() == 1);

    // product computation for correlations
    int bin;
    double const dbin = (rmax - rmin)/nBins;
    std::vector<double> disp(2, 0);
    double dist;
    for (int i=0; i < r.shape(0); i++) {
        for (int j=i; j < r.shape(0); j++) {
            if (i != j) { nPairs++; }
            disp = {r(j, 0) - r(i, 0), r(j, 1) - r(i, 1)};                  // difference vector
            if (periodic_boundary_conditions) {
                for (int dim=0; dim < 2; dim++) {
                    disp[dim] = std::remainder(disp[dim], systemSize[dim]); // wrap around periodic boundary conditions
                }
            }
            dist = std::sqrt(disp[0]*disp[0] + disp[1]*disp[1]);            // distance corresponding to difference vector
            if (dist < rmin || dist >= rmax) { continue; }                  // distance out of bins
            bin = (dist - rmin)/dbin;                                       // bin corresponding to distance
            if (values_is_1D) {
                c(bin, 1) +=                                                // product for scalars
                    (v1(i)*std::conj(v2(j))
                        + v1(j)*std::conj(v2(i)))/2.;
            }
            else {
                for (int d=0; d < v1.shape(1); d++) {
                    c(bin, 1) +=                                            // scalar product for 1D vectors
                        (v1(i, d)*std::conj(v2(j, d))
                            + v1(j, d)*std::conj(v2(i, d)))/2.;
                }
            }
            occupancy[bin] += 1;
        }
    }

    // normalise
    for (int bin=0; bin < nBins; bin++) {
        if (occupancy[bin] > 0) {
            // mean over computed values
            c(bin, 1) /= occupancy[bin];
            // correction by pair distribution function
            if (!rescale_pair_distribution) { continue; }
            if (bin == 0 && rmin == 0) { continue; }      // do not consider 0th bin
            c(bin, 1) /=
                ((double) occupancy[bin]/nPairs)            // histogram value
                *area/((rmax - rmin)/nBins)                 // normalisation
                /(2*M_PI*(rmin + bin*(rmax - rmin)/nBins)); // radial projection
        }
    }

    return correlations;
}

/*
 *  Histograms
 *
 */

pybind11::array_t<long int> getHistogram(
    pybind11::array_t<double> const& values,
    pybind11::array_t<double> const& bins) {
/*
Compute histogram from user-defined bins.
*/

    pybind11::buffer_info vBUFF = values.request();
    assert(vBUFF.ndim == 1);
    double* const vPTR = (double*) vBUFF.ptr;
    std::vector<double> v(vPTR, vPTR + vBUFF.shape[0]);
    std::sort(v.begin(), v.end());

    auto b = bins.unchecked<1>();
    assert(b.ndim() == 1);
    long int const nBins = b.shape(0) - 1;

    pybind11::array_t<long int> histogram(nBins);
    auto h = histogram.mutable_unchecked<1>();
    for (pybind11::ssize_t i=0; i < h.shape(0); i++) { h(i) = 0; }

    long int bin = 0;
    for (std::vector<double>::size_type i=0; i < v.size(); i++) {
        if (v[i] < b(0)) { continue; }
        while (v[i] >= b(bin + 1)) {
            bin++;
            if (bin == nBins) { return histogram; }
        }
        h(bin)++;
    }
    return histogram;
}

pybind11::array_t<long int> getHistogramLinear(
    pybind11::array_t<double> const& values,
    long int const& nBins, double const& vmin, double const& vmax) {
/*
Compute histogram with linearly spaced bins.
*/

    auto v = values.unchecked<1>();
    assert(v.ndim() == 1);

    pybind11::array_t<long int> histogram(nBins, 0);
    auto h = histogram.mutable_unchecked<1>();
    for (pybind11::ssize_t i=0; i < h.shape(0); i++) { h(i) = 0; }

    double const dbin = (vmax - vmin)/nBins;
    for (pybind11::ssize_t i=0; i < v.shape(0); i++) {
        if (v(i) < vmin || v(i) > vmax) { continue; }
        long int const bin = (v(i) - vmin)/dbin;
        h(bin)++;
    }
    return histogram;
}

/*
 *  Binding
 *
 */

PYBIND11_MODULE(bind, m) {

    m.def("getWaveVectors2D", &getWaveVectors2D,
        "Return wave vectors associated to rectangular box.\n"
        "\n"
        "Parameters\n"
        "----------\n"
        "L : float or (1,)- or (2,) float array-like\n"
        "    Size of the box.\n"
        "q : float\n"
        "    Target wave vector norm.\n"
        "dq : float\n"
        "    Width of wave vector norm interval. (default: 0.1)\n"
        "\n"
        "Returns\n"
        "-------\n"
        "wv : (*, 2) float numpy array\n"
        "    Array of (2\\pi/L nx, 2\\pi/L ny) wave vectors corresponding to\n"
        "    to the target interval [`q' - `dq'/2, `q' + `dq'/2].\n"
        "    NOTE: Only a single vector of each pair of opposite wave\n"
        "          vectors is returned. Here it is chosen such that ny >= 0.",
        pybind11::arg("L"),
        pybind11::arg("q"),
        pybind11::arg("dq")=0.1);

    m.def("getFT2D", &getFT2D,
        "Return 2D Fourier transform of delta-peaked values.\n"
        "\n"
        ".. math::"
        "V(k_l) = \\sum_i \\exp(-1i k_l \\cdot r_i) v_i\n"
        "\n"
        "Parameters\n"
        "----------\n"
        "positions : (*, 2) float array-like\n"
        "    Positions r_i of delta-peaked values.\n"
        "L : float or (1,)- or (2,) float array-like\n"
        "    Size of the box.\n"
        "values : (*,) complex array-like\n"
        "    Delta-peaked values v_i.\n"
        "q : float\n"
        "    Target wave vector norm.\n"
        "dq : float\n"
        "    Width of wave vector norm interval. (default: 0.1)\n"
        "\n"
        "Returns\n"
        "-------\n"
        "ft : (**,) complex numpy array\n"
        "    Fourier transform of `values' for each wave vector in the\n"
        "    target  norm interval [`q' - `dq'/2, `q' + `dq'/2].\n"
        "    NOTE: These are given by getWaveVectors2D.\n",
        pybind11::arg("positions"),
        pybind11::arg("L"),
        pybind11::arg("values"),
        pybind11::arg("q"),
        pybind11::arg("dq")=0.1);

    m.def("getRadialCorrelations2D", &getRadialCorrelations2D,
        "Compute two-dimensional (2D) radial correlations between `values1'\n"
        "(and `values2') associated to each of the positions of a 2D system\n"
        "with linear size(s) `L'.\n"
        "\n"
        ".. math::"
        "C(|r1 - r0|) = \\langle values1(r0) \\cdot values2(r1)^* \\rangle\n"
        "\n"
        "Parameters\n"
        "----------\n"
        "positions : (*, 2) float array-like\n"
        "    Positions associated to values.\n"
        "L : float or (1,)- or (2,) float array-like\n"
        "    Size of the system box in each dimension.\n"
        "values1 : (*,) or (*, **) float array-like\n"
        "    Values to compute the correlations of `values2' with.\n"
        "values2 : (*,) or (*, **) float array-like\n"
        "    Values to compute the correlations of `values1' with.\n"
        "    NOTE: if `values2' is empty then it is replaced with `values1'.\n"
        "nBins : int\n"
        "    Number of intervals of distances on which to compute the\n"
        "    correlations.\n"
        "periodic_boundary_conditions : bool\n"
        "    Use periodic boundary conditions when computing distances\n"
        "    between points. (default: True)\n"
        "rmin : float\n"
        "    Minimum distance (included) at which to compute the\n"
        "    correlations. (default: 0)\n"
        "rmax : float\n"
        "    Maximum distance (excluded) at which to compute the\n"
        "    correlations. (default: 0)\n"
        "    NOTE: if max == 0 then max = min(L)/2.\n"
        "rescale_pair_distribution : bool\n"
        "    Rescale correlations by pair distribution function.\n"
        "    (default: False)\n"
        "\n"
        "Returns\n"
        "-------\n"
        "correlations : (nBins, 2) complex Numpy array\n"
        "    Array of (r, C(r)) where r is the centre of the correlation\n"
        "    histogram bin and C(r) is the height of the correlation\n"
        "    histogram bin.",
        pybind11::arg("positions"),
        pybind11::arg("L"),
        pybind11::arg("values1"),
        pybind11::arg("values2"),
        pybind11::arg("nBins"),
        pybind11::arg("periodic_boundary_conditions")=true,
        pybind11::arg("rmin")=0,
        pybind11::arg("rmax")=0,
        pybind11::arg("rescale_pair_distribution")=false);

    m.def("getHistogram", &getHistogram,
        "Build an histogram counting the occurences of values within the\n"
        "intervals defined by bins.\n"
        "\n"
        "Parameters\n"
        "----------\n"
        "values : (*,) float array-like\n"
        "    Values to count.\n"
        "bins : (**,) float array-like\n"
        "    Limits of the bins.\n"
        "\n"
        "Returns\n"
        "-------\n"
        "histogram : (** - 1,) int numpy array\n"
        "    Histogram.",
        pybind11::arg("values"),
        pybind11::arg("bins"));

    m.def("getHistogramLinear", &getHistogramLinear,
        "Build an histogram counting the occurences of values within linear\n"
        "intervals between a minimum and a maximum value.\n"
        "\n"
        "Parameters\n"
        "----------\n"
        "values : (*,) float array-like\n"
        "    Values to count.\n"
        "nBins : int\n"
        "    Number of bins.\n"
        "vmin : float\n"
        "    Minimum value of the bins (included).\n"
        "vmax : float\n"
        "    Maximum value of the bins (excluded).\n"
        "\n"
        "Returns\n"
        "-------\n"
        "histogram : (nBins,) int numpy array\n"
        "    Histogram.",
        pybind11::arg("values"),
        pybind11::arg("nBins"),
        pybind11::arg("vmin"),
        pybind11::arg("vmax"));

}
