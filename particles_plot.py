"""
Plot simulations of particles.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import FancyArrow
from numbers import Number

class WindowClosedException(Exception): pass

def _update_canvas(fig):
    fig.canvas.draw_idle()
    fig.canvas.start_event_loop(0.001)
    if not(plt.fignum_exists(fig.number)):  # throw error when window is closed
        raise WindowClosedException

def plot_pbc(positions, L, diameters=None,
    colours=None, alpha=None, arrows=None,
    fig=None, ax=None, update=True):
    """
    Plot system with periodic boundary conditions. Particles are represented
    as circles, and arrows starting from their centre may be added.

    Parameters
    ----------
    positions : (*, 2) float array-like
        Positions of centres of particles.
    L : float or (1,) or (2,) float array-like
        System size.
    diameters : (*,) float array-like, float, or None
        Diameter(s) of circles representing particles. (default: None)
        NOTE: if diameters is None then all diameters are set to 1.
    colours : (*,) str array-like or None
        Colours to apply to circles representing particles. (default: None)
        NOTE: if colours is None then circles have black edges and transparent
              faces.
    alpha : (*,) float array-like or None
        Opacity of circles' colours. (default: None)
        NOTE: if alpha is None then opacity for all particles is set to 1.
    arrows : (*, 2) float array-like or None
        Arrows to add to circles representing particles. (default: None)
        NOTE: if arrows is None then no arrows are added.
    fig : matplotlib.figure.Figure or None
        Figure on which to plot. (default: None)
        NOTE: if fig is None then a new figure and axes subplot is created.
    ax : matplotlib.axes._subplots.AxesSubplot or None
        Axes subplot on which to plot. (default: None)
        NOTE: if ax is None then a new figure and axes subplot is created.
    update : bool
        Update figure canvas. (default: True)

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure.
    ax : matplotlib.axes._subplots.AxesSubplot
        Axes subplot.
    """

    # INITIALISE

    positions = np.array(positions)
    N = positions.shape[0]
    assert positions.shape[1] == 2
    assert positions.ndim == 2
    L = np.array(L)
    if L.ndim == 0: L = np.array([L, L])
    elif L.ndim == 1 and L.size == 1: L = np.array([L[0], L[0]])
    elif L.ndim > 1 or L.size != 2: raise ValueError("Invalid 2D system size.")
    positions = (positions + L/2)%L - L/2   # positions in [-L/2, L/2]^2

    def _set_lim(ax_, L_):
        ax_.set_xlim([-L[0]/2, L[0]/2])
        ax_.set_ylim([-L[1]/2, L[1]/2])
        ax_.set_aspect("equal")

    if type(fig) == type(None) or type(ax) == type(None):

        plt.ioff()
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 10)                                         # set figure size
        try: fig.canvas.window().setFixedSize(fig.canvas.window().size())   # set window size
        except AttributeError: pass

        # set figure limits
        _set_lim(ax, L)
        fig.canvas.mpl_connect("button_press_event",    # reset figure limits on double click
            lambda event: event.dblclick and _set_lim(ax, s))

    plt.sca(ax)
    try:
        # make zoom persistent
        fig.canvas.toolbar.push_current()
        ax.cla()
        fig.canvas.toolbar.back()
    except AttributeError:
        ax.cla()
        _set_lim(ax, L)

    # CIRCLES

    if type(diameters) is type(None):
        diameters = np.full((N,), fill_value=1, dtype=float)
    elif isinstance(diameters, Number):
        diameters = np.full((N,), fill_value=diameters, dtype=float)
    max_diameter = max(diameters)

    if type(colours) is type(None):
        fill = False
        colours = np.full((N,), fill_value="black", dtype="<U5")
    else:
        fill = True

    if type(alpha) is type(None):
        alpha = np.full((N,), fill_value=1, dtype=int)

    # orignal
    circles = list(map(
        lambda i: plt.Circle(
            positions[i], diameters[i]/2,
            color=(colours[i], alpha[i]), fill=fill),
        range(N)))
    # periodic boundary copies
    for dim in range(2):
        for minmax in range(2):
            indices = np.where(
                np.abs(positions[:, dim] - minmax*L[dim]) < max_diameter/2)[0]
            circles += list(map(
                lambda i: plt.Circle(
                    (
                        positions[i][0] + (1 - dim)*(1 - 2*minmax)*L[0],
                        positions[i][1] + dim*(1 - 2*minmax)*L[1]),
                    diameters[i]/2,
                    color="black", fill=False),
                indices))

    coll = PatchCollection(
        circles,
        edgecolors=list(map(lambda c: c.get_edgecolor(), circles)),
        facecolors=list(map(lambda c: c.get_facecolor(), circles)),
        linestyles=list(map(lambda c: c.get_linestyle(), circles)))
    ax.add_collection(coll)

    # ARROWS

    if not(type(arrows) is type(None)):

        arrows = list(map(
            lambda i: FancyArrow(*positions[i], *arrows[i],
                    width=0.1, length_includes_head=True, color="black"),
            range(N)))

        coll = PatchCollection(
            arrows,
            edgecolors=list(map(lambda a: a.get_edgecolor(), arrows)),
            facecolors=list(map(lambda a: a.get_facecolor(), arrows)))
        ax.add_collection(coll)

    # UPDATE

    if update: _update_canvas(fig)
    return fig, ax

