"""
Module plot provides objects and functions to be used in matplotlib plots.
Requires seaborn for colourmaps.
"""

import numpy as np

from collections import OrderedDict

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Slider
from matplotlib.colors import Colormap as Colourmap
from matplotlib.colors import Normalize as Normalise
from matplotlib.cm import ScalarMappable
import matplotlib as mpl
import matplotlib.tri as tr

import sys
try: import seaborn
except ModuleNotFoundError: pass

# DEFAULT VARIABLES

_markers = (                        # --- default markers (see matplotlib.lines.Line2D.filled_markers)
    "o", "^", "s", "*", "X", "D", "8", "v", "<", ">", "h", "H", "p", "d", "P")

if mpl.__version__ >= "3":
    _linestyles = (                 # --- default linestyles (see https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html)
        (0, ()),                    # solid
        (0, (5, 1)),                # densely dashed
        (0, (1, 1)),                # densely dotted
        (0, (3, 1, 1, 1)),          # densely dashdotted
        (0, (3, 1, 1, 1, 1, 1)),    # densely dashdotdotted
        (0, (5, 5)),                # dashed
        (0, (1, 5)),                # dotted
        (0, (3, 5, 1, 5)),          # dashdotted
        (0, (3, 5, 1, 5, 1, 5)),    # dashdotdotted
        (0, (5, 10)),               # loosely dashed
        (0, (1, 10)),               # loosely dotted
        (0, (3, 10, 1, 10)),        # loosely dashdotted
        (0, (3, 10, 1, 10, 1, 10))) # loosely dashdotdotted
else:
    _linestyles = (                 # --- default linestyles (see matplotlib.lines.Line2D.lineStyles)
        "-",                        # solid
        "--",                       # dashed
        "-.",                       # dash-dotted
        ":")                        # dotted

_fillstyles = (                     # --- default fillstyles (see matplotlib.lines.Line2D.fillStyles)
    "full", "none", "left", "right", "bottom", "top")

# FUNCTIONS AND CLASSES

def set_font_size(font_size):
    """
    Set matplotlib font size.

    Parameters
    ----------
    font_size : int
        Font size.
    """

    mpl.rcParams.update({"font.size": font_size})

def list_colours(value_list, colourmap="colorblind", sort=True):
    """
    Creates hash table of colours from colourmap, defined according to
    value_list index, with value_list elements as keys.

    Parameters
    ----------
    value_list : list
        List of values.
    colourmap : matplotlib.colors.Colormap (colourmap)
                or str (matplotlib or seaborn colour palette name)
        Colormap or colour palette to use. (default: "colorblind")
    sort : bool
        Sort list of values before assigning colours. (default: True)

    Returns
    -------
    colours : hash table
        Hash table of colours.
    """

    value_list = list(OrderedDict.fromkeys(value_list))
    if sort: value_list = sorted(value_list)

    try:                # matplotlib colourmap

        if issubclass(colourmap.__class__, Colourmap): cmap = colourmap # input colourmap
        else: cmap = plt.get_cmap(colourmap)                            # matplotlib colourmap
        norm = Normalise(vmin=0, vmax=len(value_list) + 1)              # normalise colourmap according to list index
        scalarMap = ScalarMappable(norm=norm, cmap=cmap)                # associates scalar to colour

        return {value_list[index]: scalarMap.to_rgba(index + 1)
            for index in range(len(value_list))}

    except ValueError:  # seaborn palette

        assert "seaborn" in sys.modules

        return {value: colour
            for value, colour in zip(
                value_list,
                seaborn.color_palette(colourmap, len(value_list)))}

def list_colormap(value_list, colormap="colorblind", sort=True):
    return list_colours(value_list, colourmap=colormap, sort=True)
list_colormap.__doc__ = list_colours.__doc__

def list_markers(value_list, marker_list=_markers, sort=True):
    """
    Creates hash table of markers from markers_list, defined according to
    value_list index, with value_list elements as keys.

    Parameters
    ----------
    value_list : list
        List of values.
    marker_list : list of matplotlib markers
        List of markers to use. (default: _markers)
    sort : bool
        Sort list of values before assigning markers. (default: True)

    Returns
    -------
    markers : hash table
        Hash table of markers.
    """

    value_list = list(OrderedDict.fromkeys(value_list))
    if sort: value_list = sorted(value_list)

    return {value_list[index]: marker_list[index%len(marker_list)]
        for index in range(len(value_list))}

def list_linestyles(value_list, linestyle_list=_linestyles, sort=True):
    """
    Creates hash table of line styles from linestyle_list, defined according to
    value_list index, with value_list elements as keys.

    Parameters
    ----------
    value_list : list
        List of values.
    linestyle_list : list of matplotlib line styles
        List of line styles to use.
        (default: _linestyles)
    sort : bool
        Sort list of values before assigning line styles. (default: True)

    Returns
    -------
    linestyles : hash table
        Hash table of line styles.
    """

    value_list = list(OrderedDict.fromkeys(value_list))
    if sort: value_list = sorted(value_list)

    return {value_list[index]: linestyle_list[index%len(linestyle_list)]
        for index in range(len(value_list))}

def list_fillstyles(value_list, fillstyle_list=_fillstyles, sort=True):
    """
    Creates hash table of fill styles from fillstyle_list, defined according to
    value_list index, with value_list elements as keys.

    Parameters
    ----------
    value_list : list
        List of values.
    fillstyle_list : list of matplotlib fill styles
        List of fill styles to use.
        (default: _fillstyles)
    sort : bool
        Sort list of values before assigning fill styles. (default: True)

    Returns
    -------
    fillstyles : hash table
        Hash table of fill styles.
    """

    value_list = list(OrderedDict.fromkeys(value_list))
    if sort: value_list = sorted(value_list)

    return {value_list[index]: fillstyle_list[index%len(fillstyle_list)]
        for index in range(len(value_list))}

def contours(x, y, z, vmin=None, vmax=None, contours=20, cmap=plt.cm.jet,
    colorbar_position="right", colorbar_orientation="vertical",
    logx=False, logy=False, logz=False):
    """
    Plot contours from 3D data.

    Parameters
    ----------
    x : (*,) float array-like
        x-axis data.
    y : (**,) or (*, **) float array-like
        y-axis data.
    z : (*, **) float array-like
        z-axis data to represent with color map.
    vmin : float or None
        Minimum value for the colorbar. (default: None)
        NOTE: if vmin == None then min(z) is taken.
    vmax : float or None
        Maximum value for the colorbar. (default: None)
        NOTE: if vmax == None then max(z) is taken.
    contours : int
        Number of contour lines. (default: 20)
        (see matplotlib.pyplot.tricontourf)
    cmap : matplotlib colorbar
        Matplotlib colorbar to be used. (default: matplotlib.pyplot.cm.jet)
    colorbar_position : string
        Position of colorbar relative to axis. (default: "right")
    colorbar_orientation : string
        Orientation of colorbar. (default: "vertical")
    logx : bool
        Logarithmically spaced x-axis data. (default: False)
    logy : bool
        Logarithmically spaced y-axis data. (default: False)
    logz : bool
        Logarithmically spaced z-axis data. (default: False)

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    colorbar : matplotlib.colorbar
        Colorbar.
    """

    x = np.array(x, dtype=float)
    assert x.ndim == 1
    y = np.array(y, dtype=float)
    if y.ndim == 1: y = np.full((x.size, y.size), fill_value=y)
    assert y.ndim == 2 and y.shape[0] == x.size
    z = np.array(z, dtype=float)
    assert z.shape == y.shape

    vmin = vmin if not(vmin is None) else z[~np.isnan(z)].min()
    vmax = vmax if not(vmax is None) else z[~np.isnan(z)].max()
    norm = Normalise(vmin=vmin, vmax=vmax)
    scalarMap = ScalarMappable(norm=norm, cmap=cmap)

    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(colorbar_position, size="5%", pad=0.05)
    colorbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm,
        orientation=colorbar_orientation)

    x, y, z = np.transpose([[x[i], y[i, j], z[i, j]]
        for i in range(x.size) for j in range(y[i].size)])
    triang = tr.Triangulation(x, y)
    triang.set_mask(np.isnan(z[triang.triangles]).any(axis=1))

    ax.tricontourf(triang, z, contours, cmap=cmap, norm=norm)

    return fig, ax, colorbar

def combine_hex_values(d):
    """
    Mix colours.
    https://stackoverflow.com/questions/61488790/how-can-i-proportionally-mix-colors-in-python

    Parameters
    ----------
    d : {colour hex string: float}
        Colours and their proportion.

    Returns
    -------
    colour : colour hex string
        Mix colour.
    """

    assert type(d) is dict
    if len(d) == 0: return

    d_items = sorted(d.items())
    tot_weight = sum(d.values())
    red = int(sum([int(k[:2], 16)*v for k, v in d_items])/tot_weight)
    green = int(sum([int(k[2:4], 16)*v for k, v in d_items])/tot_weight)
    blue = int(sum([int(k[4:6], 16)*v for k, v in d_items])/tot_weight)
    zpad = lambda x: x if len(x)==2 else "0" + x

    return zpad(hex(red)[2:]) + zpad(hex(green)[2:]) + zpad(hex(blue)[2:])

class FittingLine:
    """
    Provided a matplotlib.axes.Axes object, this object:
    > draws a staight line on the corresponding figure, either in a log-log
    (powerlaw fit), lin-log (exponential fit), log-lin (logarithmic fit), or a
    lin-lin (linear fit) plot,
    > displays underneath the figure a slider which controls the slope of the
    line, the slider can be expanded / shrinked by scrolling,
    > shows fitting line expression in legend.

    Clicking on the figure updates the position of the line such that it passes
    through the clicked point.

    Attributes (partial list)
    ----------

    FittingLine.ax : matplotlib.axes.Axes object
        Plot Axes object.

    FittingLine.x_fit : string
        x-data name in legend.
    FittingLine.y_fit : string
        y-data name in legend.
    FittingLine.color : any matplotlib color
        Color of fitting line.
    FittingLine.linestyle : any matplotlib linestyle
        Linestyle of fitting line.

    FittingLine.x0 : float
        x-coordinate of clicked point.
    FittingLine.y0 : float
        y-coordinate of clicked point.
    FittingLine.slope : float
        Slope of fitting line.

    FittingLine.line : matplotlib.lines.Line2D object
        Line2D representing fitting line.

    FittingLine.slider : matplotlib Slider widget
        Slope slider.

    FittingLine.law : string
        Fitting line law.
    FittingLine.func : function
        Fitting line function.
    """

    def __init__(self, ax, slope, slope_min=None, slope_max=None,
        color="black", linestyle="--", slider=True, output=True,
        legend=False, exp_format="{:.2e}", font_size=None,
        legend_frame=False, handlelength=None, **kwargs):
        """
        Parameters
        ----------
        ax : matplotlib.axes.Axes object
            Axes object on which to draw fitting line.
        slope : float
            Initial slope of fitting line in log-log plot.
        slope_min : float
            Minimum slope of fitting line for slider. (default: None)
            NOTE: if slope_min == None, then slope_min is taken to be slope.
        slope_max : float
            Maximum slope of fitting line for slider. (default: None)
            NOTE: if slope_max == None, then slope_max is taken to be slope.
        color : any matplotlib color
            Color of fitting line. (default: "black")
        linestyle : any matplotlib line style
            Line style of fitting line. (default: "--")
        slider : bool
            Display slider for slope. (default: True)
        output : bool
            Print coordinates of clicked point and slope at each update.
            (default: True)
        legend : bool
            Display legend. (default: True)
        exp_format : string
            Exponent string format in legend. (default: {:.2e})
            NOTE: Only if legend == True.
        font_size : float
            Legend font size. (default: None)
            NOTE: if font_size == None, the font size is not imposed.
        legend_frame : bool
            Display legend frame. (default: True)
        handlelength : float
            Horizontal line length in legend. (default: None)

        Optional keyword arguments
        --------------------------
        x_fit : string
            Custom name of x data for fitting line expression in legend.
        y_fit : string
            Custom name of y data for fitting line expression in legend.
        """

        self.ax = ax        # Axes object
        plt.sca(self.ax)    # set current axis

        self.x_fit = (kwargs["x_fit"] if "x_fit" in kwargs
            else self.ax.get_xlabel()).replace("$", "") # x data name in legend
        self.y_fit = (kwargs["y_fit"] if "y_fit" in kwargs
            else self.ax.get_ylabel()).replace("$", "") # y data name in legend
        self.color = color                              # color of fitting line
        self.linestyle = linestyle                      # linestyle of fitting line

        self.slope = slope  # slope of fitting line

        self.ax.set_xlim(self.ax.get_xlim())
        self.ax.set_ylim(self.ax.get_ylim())
        xscale, yscale = self.ax.get_xscale(), self.ax.get_yscale()
        self.x0 = ( # x-coordinate of clicked point set as middle of graph
            np.exp(np.ma.log(self.ax.get_xlim()).mean()) if xscale == "log"
            else np.mean(self.ax.get_xlim()))
        self.y0 = ( # y-coordinate of clicked point set as middle of graph
            np.exp(np.ma.log(self.ax.get_ylim()).mean()) if yscale == "log"
            else np.mean(self.ax.get_ylim()))
        if xscale == "log" and yscale == "log":
            self.law = "powerlaw"
            self.func = _powerlaw
            self.label = r"$%s \propto %s^{%s}$"
        elif xscale == "linear" and yscale == "log":
            self.law = "exponential"
            self.func = _exponential
            self.label = r"$%s \propto e^{%s%s}$"
        elif xscale == "log" and yscale == "linear":
            self.law = "logarithmic"
            self.func = _logarithmic
            self.label = r"$%s \propto %s\log(%s)$"
        elif xscale == "linear" and yscale == "linear":
            self.law = "linear"
            self.func = _linear
            self.label = r"$%s \propto %s%s$"
        else:
            raise ValueError("Axis scales are not supported.")

        self.line, = self.ax.plot([], [], label=" ",
            color=self.color, linestyle=self.linestyle) # Line2D representing fitting line

        self.display_legend = legend                                # display legend
        if self.display_legend:
            self.x_legend = self.x0                                 # x-coordinate of fitting line legend
            self.y_legend = self.y0                                 # y-coordinate of fitting line legend
            self.legend = plt.legend(handles=[self.line], loc=10,
                bbox_to_anchor=(self.x_legend, self.y_legend),
                bbox_transform=self.ax.transData,
                frameon=legend_frame, handlelength=handlelength)    # fitting line legend
            self._set_fontsize(font_size)                           # set legend font size
            self.legend_artist = self.ax.add_artist(self.legend)    # fitting line legend artist object
            self.legend_artist.set_picker(10)                       # epsilon tolerance in points to fire pick event
        self.on_legend = False                                      # has the mouse been clicked on fitting line legend
        self.exp_format = exp_format                                # exponent string format in legend

        self.display_slider = slider                        # display slider
        if slope_min is None and slope_max is None: self.display_slider = False
        if self.display_slider:
            self.slider_ax = make_axes_locatable(self.ax).append_axes(
                "bottom", size="5%", pad=0.6)               # slider Axes
            self.slider = Slider(self.slider_ax, "",        # slider
                slope_min if not(slope_min is None) else slope,
                slope_max if not(slope_max is None) else slope,
                valinit=slope, initcolor='none', color="#e85e8a",
                valfmt="slope=%.3e")
            self.slider.on_changed(self.update_slope)       # call self.update_slope when slider value is changed
            self.slider.valtext.set_position((0.5, -0.75))  # centre value below slider
            self.slider.valtext.set_verticalalignment("center")
            self.slider.valtext.set_horizontalalignment("center")

        self.output = output    # print (x0, y0) and slope at each update

        self.cid_click = self.line.figure.canvas.mpl_connect(
            "button_press_event", self._on_click)       # call on click on figure
        self.cid_pick = self.line.figure.canvas.mpl_connect(
            "pick_event", self._on_pick)                # call on artist pick on figure
        self.cid_release = self.line.figure.canvas.mpl_connect(
            "button_release_event", self._on_release)   # call on release on figure
        self.cid_scroll = self.line.figure.canvas.mpl_connect(
            "scroll_event", self._on_scroll)            # call on scroll

        self.update_slope() # draw everything

    def _set_fontsize(self, font_size):
        """
        Set legend font size.

        Parameters
        ----------
        font_size : float
            Legend font size.
            NOTE: if font_size=None, the font size is not changed.
        """

        self.font_size = font_size
        if not(self.font_size is None):
            self.legend.get_texts()[0].set_fontsize(self.font_size) # set legend font size

    def _on_click(self, event):
        """
        Executes on click.

        Double click switches between powerlaw and exponential laws and updates
        figure.
        Simple click makes fitting line pass through clicked point and updates
        figure.
        """

        if event.inaxes != self.ax: # if Axes instance mouse is over is different than figure Axes
            return

        elif self.on_legend:        # if fitting line legend is being dragged
            return

        else:
            self.x0 = event.xdata   # x coordinate of clicked point
            self.y0 = event.ydata   # y coordinate of clicked point
            self.draw()             # update figure

    def _on_pick(self, event):
        """
        Executes on picking.

        Fitting line legend can be moved if dragged.
        """

        if self.display_legend == False: return

        if event.artist == self.legend_artist:  # if fitting line legend is clicked
            self.on_legend = True               # fitting line legend has been clicked

    def _on_release(self, event):
        """
        Executes on release.

        Moves fitting line legend to release position.
        """

        if self.display_legend == False: return

        if not(self.on_legend): return      # if fitting line legend has not been clicked
        self.x_legend = event.xdata         # x coordinate of fitting line legend
        self.y_legend = event.ydata         # y coordinate of fitting line legend
        self.legend.set_bbox_to_anchor(bbox=(self.x_legend, self.y_legend),
            transform=self.ax.transData)    # move legend to release point
        self.line.figure.canvas.draw()      # updates legend
        self.on_legend = False              # fitting line legend has been released

    def _on_scroll(self, event):
        """
        Executes on scroll.

        Expand/shrink slider if slider is visible.
        """

        if not(self.display_slider): return

        dilation = 0.1                                          # dilation parameter
        factor = 1 + dilation*(1 if event.button == "up" else -1)

        middle = (self.slider.valmin + self.slider.valmax)/2    # dilate with respect to middle
        vmin = middle - (middle - self.slider.valmin)*factor
        self.slider.valmin = vmin
        vmax = middle - (middle - self.slider.valmax)*factor
        self.slider.valmax = vmax
        if self.output:
            print("valmin = %s; valmax = %s; \b" % (vmin, vmax))

        self.slider_ax.set_xlim([vmin, vmax])
        if self.slider.val < vmin or self.slider.val > vmax:
            if self.slider.val < vmin: self.slider.set_val(vmin)
            if self.slider.val > vmax: self.slider.set_val(vmax)
        verts = self.slider.poly.xy                             # update slider filler
        verts[0] = verts[4] = self.slider.valmin, .25
        verts[1] = self.slider.valmin, .75
        verts[2] = self.slider.val, .75
        verts[3] = self.slider.val, .25

        output = self.output
        self.output = False
        self.update_slope()                                     # update slope, legend, and figure
        self.output = output

    def update_slope(self, val=None):
        """
        Set fitting line slope according to slider value and updates figure.
        """

        if self.display_slider: self.slope = self.slider.val    # updates slope of fitting line
        self.update_legend()                                    # updates legend and figure

    def update_legend(self):
        """
        Updates fitting line legend.
        """

        if self.law == "powerlaw":
            self.line.set_label(r"$%s \sim %s^{%s}$" % (self.y_fit,
                self.x_fit, self.exp_format.format(self.slope)))        # fitting line label
        elif self.law == "exponential":
            self.line.set_label(r"$%s \sim e^{%s \times %s}$" % (self.y_fit,
                self.exp_format.format(self.slope), self.x_fit))        # fitting line label
        elif self.law == "logarithmic":
            self.line.set_label(r"$%s \sim %s \log(%s)}$" % (self.y_fit,
                self.exp_format.format(self.slope), self.x_fit))        # fitting line label
        elif self.law == "linear":
            self.line.set_label(r"$%s \sim %s \times %s$" % (self.y_fit,
                self.exp_format.format(self.slope), self.x_fit))        # fitting line label

        self.line.set_label(self.label                                  # fitting line label
            % (self.y_fit, self.exp_format.format(self.slope), self.x_fit))

        if self.display_legend == True:
            self.legend.get_texts()[0].set_text(self.line.get_label())  # updates fitting line legend
        self.draw()                                                     # updates figure

    def draw(self):
        """
        Updates figure with desired fitting line.
        """

        if self.output:
            intercept = self.func(self.x0, self.y0, self.slope,
                0 if self.ax.get_xscale() == "linear" else 1)
            print("x0 = %s; y0 = %s; slope = %s; intercept = %s"
                % (self.x0, self.y0, self.slope, intercept))

        self.line.set_data(self.ax.get_xlim(), list(map(
            lambda x: self.func(self.x0, self.y0, self.slope, x),
            self.ax.get_xlim())))       # line passes through clicked point according to law
        self.line.figure.canvas.draw()  # updates figure

def _powerlaw(x0, y0, slope, x):
    """
    From point (x0, y0) and parameter slope, returns y = f(x) such that:
    > f(x) = a * (x ** slope)
    > f(x0) = y0

    Parameters
    ----------
    x0, y0, slope, x : float

    Returns
    -------
    y = f(x) : float
    """

    return y0 * ((x/x0) ** slope)

def _exponential(x0, y0, slope, x):
    """
    From point (x0, y0) and parameter slope, returns y = f(x) such that:
    > f(x) = a * exp(x * slope)
    > f(x0) = y0

    Parameters
    ----------
    x0, y0, slope, x : float

    Returns
    -------
    y = f(x) : float
    """

    return y0 * np.exp((x - x0) * slope)

def _logarithmic(x0, y0, slope, x):
    """
    From point (x0, y0) and parameter slope, returns y = f(x) such that:
    > f(x) = log(x) * slope + a
    > f(x0) = y0

    Parameters
    ----------
    x0, y0, slope, x : float

    Returns
    -------
    y = f(x) : float
    """

    return np.log(x/x0) * slope + y0

def _linear(x0, y0, slope, x):
    """
    From point (x0, y0) and parameter slope, returns y = f(x) such that:
    > f(x) = x * slope + a
    > f(x0) = y0

    Parameters
    ----------
    x0, y0, slope, x : float

    Returns
    -------
    y = f(x) : float
    """

    return (x - x0) * slope + y0

