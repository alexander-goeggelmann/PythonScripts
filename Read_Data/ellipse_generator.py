import os
import matplotlib
import bokeh
import sys
import warnings
import gc

import numpy as np
import matplotlib.pyplot as plt
import holoviews as hv
import pandas as pd
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.stats import ncx2, beta
from scipy.special import comb
from IPython.display import clear_output

from holoviews.operation.datashader import datashade

import global_parameters as gl
import pixel_day_generator as pdg
import pulse_generator as pg
import calibration_generator as cg

sys.path.append(gl.PATH_TO_PLOTTING)
from TexToUni import tex_to_uni
import PlottingTool as ptool
hv.extension('bokeh', logo=False)

# TODO: Generate a function, which creates/loads and returns the list of pixels.


def get_lines(xaxis, M=True):
    """ Get the entries with energies corresponding to either the M1 or N1 lines
        of the Ho-163 EC spectrum.

    Args:
        xaxis (numpy.array): List of energies.
        M (bool, optional): True if entries with energies similar to the M1 line
                            False if entries with energies similar to the N1
                            line should be returned. Defaults to True.

    Returns:
        numpy.array: A list of booleanas giving the entries of corresponding events.
    """

    if M:
        return (xaxis > 2040) & (xaxis < 2060)
    else:
        return (xaxis > 410) & (xaxis < 430)


def rotate(xaxis, yaxis, rot=False, **pars):
    """ Find the principle components of the xaxis-yaxis coordinate system.

    Args:
        xaxis (numpy.array): X-values.
        yaxis (numpy.array): Y-values.
        rot (bool, optional): True if one of the principle components should be
                              returned. False if xaxis / yaxis should be
                              returned. Defaults to False.

    Returns:
        numpy.array: A principle component axis of the parameter space.
    """

    # Check if the principle components should be determined.
    if not rot:
        return xaxis / yaxis

    # Check if the rotation angle is known. If not, it has to be calculated.
    if not pars:
        alphas = get_rotate_pars(xaxis, yaxis)
    else:
        alphas = pars

    # Calculate the principle component axis.
    new_y = np.sin(alphas["alpha0"]) * xaxis - np.cos(alphas["alpha0"]) * yaxis
    # Get the angle between new_y and xaxis. A value of 0 is expected.
    alpha = alphas["alpha1"]

    return np.sin(alpha) * xaxis - np.cos(alpha) * new_y


def get_rotate_pars(xaxis, yaxis):
    """ Find the angle between the two axis and the angle between the principle
        component axis xaxis..

    Args:
        xaxis (numpy.array): X-values.
        yaxis (numpy.array): Y-values.

    Returns:
        dict: alpha0: The angle between xaxis and yaxis. alpha1: The angle
              between xaxis and the principle component axis.
    """

    # Identify events of the Ho-163 EC N1 line.
    cut_N = get_lines(xaxis, M=False)
    events_N = yaxis[cut_N].mean()
    # Identify events of the M1 line.
    cut_M = get_lines(xaxis, M=True)
    events_M = yaxis[cut_M].mean()

    # Energy difference between the N1 and M1 lines.
    DIFF_E = 2050 - 420

    # Get the gradient for the pulse shape parameter
    a = (events_M - events_N) / DIFF_E
    # Get the angle of the slope.
    alpha0 = np.arctan(a)

    # Calculate the principle component axis.
    new_y = np.sin(alpha) * xaxis - np.cos(alpha) * yaxis

    # Repeat the steps from above for the principle component axis.
    # An alpha of 0 is expected.
    events_N = new_y[cut_N].mean()
    events_M = new_y[cut_M].mean()
    alpha1 = np.arctan((new_y[cut_M].mean() - new_y[cut_N].mean()) / DIFF_E)

    return {"alpha0":alpha0, "alpha1":alpha1}


def get_dependency(data, column0, column1, caldata=None, pca=False):
    """ Get the dependency between two pulse shape parameters of the data.

    Args:
        data (pandas.DataFrame): The data frame.
        column0 (string): The name of the first pulse shape parameter.
        column1 (string): The name of the second pulse shape parameter.
        caldata (pandas.DataFrame, optional): Data used for calibration.
                                              Defaults to None.
        pca (bool, optional): False if the fraction of the two pulse shape
                              parameters should be returned. True if a principle
                              component axis should be determined. Defaults to False.

    Returns:
        numpy.array: The dependency of the two pulse shape parameters for each entry.
    """

    # Calculate the rotaion angles between the two axis for the calibration data,
    # if calibration data should be used.
    pars = {}
    if caldata is not None:
        pars = get_rotate_pars(caldata[column0], caldata[column1])

    return rotate(data[column0], data[column1], rot=pca, **pars)


def DTR(data, caldata=None, pca=False):
    """ Get the dependency between the derivative and template fit
        pulse shape parameters.

    Args:
        data (pandas.DataFrame): The data frame.
        caldata (pandas.DataFrame, optional): Data used for calibration.
                                              Defaults to None.
        pca (bool, optional): False if the fraction of the two pulse shape
                              parameters should be returned. True if a principle
                              component axis should be determined. Defaults to False.

    Returns:
        numpy.array: The dependency of the two pulse shape parameters for each entry.
    """

    return get_dependency(data, gl.COLUMN_DERIVATIVE_AMP,
                          gl.COLUMN_TEMPLATE_AMP, caldata=caldata, pca=pca)

def TDR(data, caldata=None, pca=False):
    """ Get the dependency between the template fit and derivative
        pulse shape parameters.

    Args:
        data (pandas.DataFrame): The data frame.
        caldata (pandas.DataFrame, optional): Data used for calibration.
                                              Defaults to None.
        pca (bool, optional): False if the fraction of the two pulse shape
                              parameters should be returned. True if a principle
                              component axis should be determined. Defaults to False.

    Returns:
        numpy.array: The dependency of the two pulse shape parameters for each entry.
    """

    return get_dependency(data, gl.COLUMN_TEMPLATE_AMP,
                          gl.COLUMN_DERIVATIVE_AMP, caldata=caldata, pca=pca)


def DFR(data, caldata=None, pca=False):
    """ Get the dependency between the derivative and matched filter
        pulse shape parameters.

    Args:
        data (pandas.DataFrame): The data frame.
        caldata (pandas.DataFrame, optional): Data used for calibration.
                                              Defaults to None.
        pca (bool, optional): False if the fraction of the two pulse shape
                              parameters should be returned. True if a principle
                              component axis should be determined. Defaults to False.

    Returns:
        numpy.array: The dependency of the two pulse shape parameters for each entry.
    """

    return get_dependency(data, gl.COLUMN_DERIVATIVE_AMP,
                          gl.COLUMN_FILTER_AMP, caldata=caldata, pca=pca)


def FDR(data, caldata=None, pca=False):
    """ Get the dependency between the matched filter and derivative
        pulse shape parameters.

    Args:
        data (pandas.DataFrame): The data frame.
        caldata (pandas.DataFrame, optional): Data used for calibration.
                                              Defaults to None.
        pca (bool, optional): False if the fraction of the two pulse shape
                              parameters should be returned. True if a principle
                              component axis should be determined. Defaults to False.

    Returns:
        numpy.array: The dependency of the two pulse shape parameters for each entry.
    """

    return get_dependency(data, gl.COLUMN_FILTER_AMP,
                          gl.COLUMN_DERIVATIVE_AMP, caldata=caldata, pca=pca)


def DIR(data, caldata=None, pca=False):
    """ Get the dependency between the derivative and integral
        pulse shape parameters.

    Args:
        data (pandas.DataFrame): The data frame.
        caldata (pandas.DataFrame, optional): Data used for calibration.
                                              Defaults to None.
        pca (bool, optional): False if the fraction of the two pulse shape
                              parameters should be returned. True if a principle
                              component axis should be determined. Defaults to False.

    Returns:
        numpy.array: The dependency of the two pulse shape parameters for each entry.
    """

    return get_dependency(data, gl.COLUMN_DERIVATIVE_AMP,
                          gl.COLUMN_FULL_INTEGRAL, caldata=caldata, pca=pca)


def IDR(data, caldata=None, pca=False):
    """ Get the dependency between the integral and derivative
        pulse shape parameters.

    Args:
        data (pandas.DataFrame): The data frame.
        caldata (pandas.DataFrame, optional): Data used for calibration.
                                              Defaults to None.
        pca (bool, optional): False if the fraction of the two pulse shape
                              parameters should be returned. True if a principle
                              component axis should be determined. Defaults to False.

    Returns:
        numpy.array: The dependency of the two pulse shape parameters for each entry.
    """

    return get_dependency(data, gl.COLUMN_FULL_INTEGRAL,
                          gl.COLUMN_DERIVATIVE_AMP, caldata=caldata, pca=pca)


def TFR(data, caldata=None, pca=False):
    """ Get the dependency between the template fit and matched filter
        pulse shape parameters.

    Args:
        data (pandas.DataFrame): The data frame.
        caldata (pandas.DataFrame, optional): Data used for calibration.
                                              Defaults to None.
        pca (bool, optional): False if the fraction of the two pulse shape
                              parameters should be returned. True if a principle
                              component axis should be determined. Defaults to False.

    Returns:
        numpy.array: The dependency of the two pulse shape parameters for each entry.
    """

    return get_dependency(data, gl.COLUMN_TEMPLATE_AMP,
                          gl.COLUMN_FILTER_AMP, caldata=caldata, pca=pca)


def FTR(data, caldata=None, pca=False):
    """ Get the dependency between the matched filter and template fit
        pulse shape parameters.

    Args:
        data (pandas.DataFrame): The data frame.
        caldata (pandas.DataFrame, optional): Data used for calibration.
                                              Defaults to None.
        pca (bool, optional): False if the fraction of the two pulse shape
                              parameters should be returned. True if a principle
                              component axis should be determined. Defaults to False.

    Returns:
        numpy.array: The dependency of the two pulse shape parameters for each entry.
    """

    return get_dependency(data, gl.COLUMN_FILTER_AMP,
                          gl.COLUMN_TEMPLATE_AMP, caldata=caldata, pca=pca)


def TIR(data, caldata=None, pca=False):
    """ Get the dependency between the template fit and integral
        pulse shape parameters.

    Args:
        data (pandas.DataFrame): The data frame.
        caldata (pandas.DataFrame, optional): Data used for calibration.
                                              Defaults to None.
        pca (bool, optional): False if the fraction of the two pulse shape
                              parameters should be returned. True if a principle
                              component axis should be determined. Defaults to False.

    Returns:
        numpy.array: The dependency of the two pulse shape parameters for each entry.
    """

    return get_dependency(data, gl.COLUMN_TEMPLATE_AMP,
                          gl.COLUMN_FULL_INTEGRAL, caldata=caldata, pca=pca)


def ITR(data, caldata=None, pca=False):
    """ Get the dependency between the integral and template fit
        pulse shape parameters.

    Args:
        data (pandas.DataFrame): The data frame.
        caldata (pandas.DataFrame, optional): Data used for calibration.
                                              Defaults to None.
        pca (bool, optional): False if the fraction of the two pulse shape
                              parameters should be returned. True if a principle
                              component axis should be determined. Defaults to False.

    Returns:
        numpy.array: The dependency of the two pulse shape parameters for each entry.
    """

    return get_dependency(data, gl.COLUMN_FULL_INTEGRAL,
                          gl.COLUMN_TEMPLATE_AMP, caldata=caldata, pca=pca)


def FIR(data, caldata=None, pca=False):
    """ Get the dependency between the matched filter and integral
        pulse shape parameters.

    Args:
        data (pandas.DataFrame): The data frame.
        caldata (pandas.DataFrame, optional): Data used for calibration.
                                              Defaults to None.
        pca (bool, optional): False if the fraction of the two pulse shape
                              parameters should be returned. True if a principle
                              component axis should be determined. Defaults to False.

    Returns:
        numpy.array: The dependency of the two pulse shape parameters for each entry.
    """

    return get_dependency(data, gl.COLUMN_FILTER_AMP,
                          gl.COLUMN_FULL_INTEGRAL, caldata=caldata, pca=pca)


def IFR(data, caldata=None, pca=False):
    """ Get the dependency between the integral and matched filter
        pulse shape parameters.

    Args:
        data (pandas.DataFrame): The data frame.
        caldata (pandas.DataFrame, optional): Data used for calibration.
                                              Defaults to None.
        pca (bool, optional): False if the fraction of the two pulse shape
                              parameters should be returned. True if a principle
                              component axis should be determined. Defaults to False.

    Returns:
        numpy.array: The dependency of the two pulse shape parameters for each entry.
    """

    return get_dependency(data, gl.COLUMN_FULL_INTEGRAL,
                          gl.COLUMN_FILTER_AMP, caldata=caldata, pca=pca)


# TODO: Could be removed.
def CHI(data, caldata=None):
    """ Get the reduced chi2 values.

    Args:
        data (pandas.DataFrame): The data frame.
        caldata (--, optional): Not used.

    Returns:
        numpy.array: The chi2 values for each entry.
    """

    return data[gl.COLUMN_TEMPLATE_CHI]


def get_ratios(data, caldata=None):
    """ Get the dependencies between each pulse shape parameters.

    Args:
        data (pandas.DataFrame): The data frame.
        caldata (pandas.DataFrame, optional): Data used for calibration.
                                              Defaults to None.

    Returns:
        dict: A dictionary containing all dependencies.
    """

    out = {}

    out["DTR"] = DTR(data, caldata=caldata)
    out["DFR"] = DFR(data, caldata=caldata)
    out["DIR"] = DIR(data, caldata=caldata)

    out["TDR"] = TDR(data, caldata=caldata)
    out["TFR"] = TFR(data, caldata=caldata)
    out["TIR"] = TIR(data, caldata=caldata)

    out["FDR"] = FDR(data, caldata=caldata)
    out["FTR"] = FTR(data, caldata=caldata)
    out["FIR"] = FIR(data, caldata=caldata)

    out["IDR"] = IDR(data, caldata=caldata)
    out["ITR"] = ITR(data, caldata=caldata)
    out["IFR"] = IFR(data, caldata=caldata)
    return out

# TODO: Could be removed.
# Label for the reduced chi2 values
# CHI_LABEL = tex_to_uni("\chi^{2} dof^{-1}")

# Dependencies of pulse shape parameters used for the x-axis.
X_FUNCS = [DTR, DFR, DIR,
           DTR, DTR, DTR, DTR,
           DFR, DFR, DFR,
           DIR, DIR,
           TFR, TFR,
           TIR][:3]

X_LABELS = ["DTR", "DFR", "DIR",
            "DTR", "DTR", "DTR", "DTR",
            "DFR", "DFR", "DFR",
            "DIR", "DIR",
            "TFR", "TFR",
            "TIR"][:3]

# Dependencies of pulse shape parameters used for the y-axis.
Y_FUNCS = [FIR, TIR, TFR,
           TIR, TFR, DFR, DIR,
           TFR, FIR, DIR,
           FIR, TIR,
           FIR, TIR,
           FIR][:3]

Y_LABELS = ["FIR", "TIR", "TFR",
            "TIR", "TFR", "DFR", "DIR",
            "TFR", "FIR", "DIR",
            "FIR", "TIR",
            "FIR", "TIR",
            "FIR"][:3]


def get_max_area(level, contours, run=5, thr=0.1, counts=5):
    """ Finds the edges of the level-th contour of contours.

    Args:
        level (int): The i-th contour of contours, which should be analyzed.
        contours (pyplot.contours): A list of contours.
        run (int, optional): A higher number means a higher accuracy. Defaults to 5.
        thr (float, optional): Defines the distance threshold of a segment of
                               the contour to the center of the contour. If the
                               distance is larger, the segment will be ignored.
                               Defaults to 0.1.
        counts (int, optional): Defines the minimum size of a segment of the
                                contour. If the size is smaller, the segment
                                will be ignored. Defaults to 5.

    Returns:
        numpy.ndarray: A 2d array of the outer points of the contour.
    """

    if contours is None:
        return None

    # TODO: Initialize these arrays with a fixed length of len(contours.allsegs[level])
    # Initialize the center of the contour to (0, 0).
    centers = np.array([np.array([0., 0.])])
    # Initialize an array, which will define if a group/segment should be used
    # to determine the edge of the contour.
    use = np.array([True])
    # Initialize the weights of a segment. If will contain the number of points
    # of each segment.
    weights = np.array([0.])

    # Loop over all segments of the contour.
    for seg in contours.allsegs[level]:
        # Add an entry to use.
        use = np.append(use, False)

        # The coordinates of the center of mass of the segment.
        x = 0.
        y = 0.

        # The number of points in the current segment.
        num_seg = seg.shape[0]

        # Checks if the size (number of points in the segment) is large enough.
        if num_seg > counts:
            use[-1] = True

            # Calculate the center of mass of the current segment.
            # Iterate over all points of the segment.
            for p in seg:
                x += p[0]
                y += p[1]
            x /= num_seg
            y /= num_seg

        weights = np.append(weights, num_seg)
        centers = np.append(centers, np.array([np.array([x, y])]), axis=0)

    # TODO: Not necessary if above TODO is done.
    # Drop the initialzing values.
    centers = centers[1:]
    use = use[1:]
    weights = weights[1:]

    # Check if each segment consists of enough points. The threshold is defined
    # by 10 % of the largest segment.
    for i, weight in enumerate(weights):
        if weight < 0.1 * weights.max():
            use[i] = False

    # Determine the center of mass of all points in the contour.
    # In each iteration the points with the largest distance to the center of
    # mass are dropped.
    for i in range(run):
        # The coordinates of the center of mass.
        center_x = 0.
        center_y = 0.

        # Normalize the (used) weights to the total number of points.
        tmp_weights = weights[use] / weights[use].sum()

        # Determine the center of mass.
        for k, center in enumerate(centers[use]):
            center_x += center[0] * tmp_weights[k]
            center_y += center[1] * tmp_weights[k]
        global_r = np.sqrt(center_x**2 + center_y**2)

        # Check for each segment, if the distance of the
        # segment to the center of mass is smaller than the threshold. If the
        # distance is larger, the segment will be ignored in the next iteration.
        for entry, center in enumerate(centers):
            if use[entry]:
                entry_r = np.sqrt(
                    (center[0] - center_x)**2 + (center[1] - center_y)**2)

                if entry_r / global_r > thr:
                    # The distance is to large.
                    use[entry] = False

        # Recalculate the center of mass. Note that there are segments,
        # which were dropped since the last calculation.
        center_x = 0.
        center_y = 0.
        total_counts = weights[use].sum()
        median_counts = np.median(weights[use])
        tmp_weights = weights[use] / total_counts

        for k, center in enumerate(centers[use]):
            center_x += center[0] * tmp_weights[k]
            center_y += center[1] * tmp_weights[k]

        # Due to the calibration, it is expected that the center of mass
        # is at (1, 1).
        dis_to_one = np.sqrt((center_x - 1.)**2 + (center_y - 1.)**2)

        # In the following it will be checked how the center of mass vary by
        # dropping single segments. If the distance of the center of mass to
        # (1, 1) is smaller by dropping a segment. This segment will be ignored
        # in the following.

        ignore_list = []
        for entry, weight in enumerate(weights[use]):
            center_x0 = 0
            center_y0 = 0
            # The number of counts in the current segment has to be removed
            # from the total number of points.
            tmp_weights = weights[use] / (total_counts - weight)

            # Calculate the temporay center of mass.
            for k, center in enumerate(centers[use]):
                if k == entry:
                    # This segment should be tested.
                    continue
                center_x0 += center[0] * tmp_weights[k]
                center_y0 += center[1] * tmp_weights[k]

            new_dis = np.sqrt((center_x0 - 1.)**2 + (center_y0 - 1.)**2)

            if (new_dis > dis_to_one) and (weight < median_counts):
                # The center of mass without the current segment is closer to
                # (1, 1) and the number of points in this segment contributes
                # only minor to the total number of points.
                ignore_list.append(entry)


    # If only at maximum three segments survive, only one of them will be used
    # to determine the edge of the contour.
    if (use.sum() - len(ignore_list)) <= 3:
        entries = None

        # Identify entries, which are not ignored.
        for entry in range(use.sum()):
            if not entry in ignore_list:
                # Determine the distance of the center of mass of
                # the segment to (1, 1).
                center = centers[use][entry]
                tmp_dis = np.sqrt((center[0] - 1.)**2 + (center[1] - 1.)**2)
                if entries is None:
                    entries = np.array([entry])
                    distances = np.array([tmp_dis])
                else:
                    entries = np.append(entries, entry)
                    distances = np.append(distances, tmp_dis)

        # Only use the segment with the smallest distance.
        for entry in entries[distances > distances.min()]:
            ignore_list.append(entry)

    # Collect all points of segments, which should be used.
    out = contours.allsegs[level][0]
    first_e = out.shape[0]

    for entry, seg in enumerate(contours.allsegs[level][use]):
        if not entry in ignore_list:
            out = np.append(out, seg[entry], axis=0)
    return out[first_e:]


def get_density_data(xdata, ydata, limit=True):
    """ Generates a density mesh from the data points.

    Args:
        xdata (numpy.array): X-values.
        ydata (numpy.array): Y-values.
        limit (bool, optional): Defines if the axes should be limited to a
                              fixed range. Defaults to True.

    Returns:
        numpy.ndarray, numpy.ndarray, numpy.ndarray: The x, y, and z meshes.
    """

    # Define limits for the axes.
    if limit:
        x_lim = (0.6, 1.4)
        y_lim = x_lim
    else:
        x_lim = (xdata.min(), xdata.max())
        y_lim = (ydata.min(), ydata.max())

    # Generate the density plot.
    h_map = datashade(hv.Points((xdata, ydata)), cmap="gray",
                      dynamic=False).redim.range(x=x_lim, y=y_lim)
    # Get the axis of three dimensions.
    x = h_map.table()["x"]
    y = h_map.table()["y"]
    z = h_map.table()["A"]

    # Define a own color map.
    # TODO: According to the warnings, this has to be changed for future
    # versions of matplotlib.
    my_palette = [(0, 0, 0)]
    for i in bokeh.palettes.Spectral11:
        my_palette.append(matplotlib.colors.hex2color(i))
    my_palette = my_palette[1:]
    my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "", my_palette)

    # Generate a mesh from the density plot.
    triang = matplotlib.tri.Triangulation(x, y)
    interpolator = matplotlib.tri.LinearTriInterpolator(triang, z)
    x_axis = np.linspace(x.min(), x.max(), 500)
    y_axis = np.linspace(y.min(), y.max(), 500)
    xx, yy = np.meshgrid(x_axis, y_axis)
    zz = interpolator(xx, yy)

    return xx, yy, zz


def get_contours(
    xdata, ydata, levels=4, thr=0.1, run=5, ellipse=True,
        color="red", counts=5, xlabel="DTR", ylabel="TFR", plot=True,
        pixel=0, path="."):
    """ Determines the density contours of the given parameter space.

    Args:
        xdata (numpy.array): X-values.
        ydata (numpy.array): Y-values.
        levels (int, optional): The number of density levels. Defaults to 4.
        thr (float, optional): Defines the distance threshold of a segment of
                               the contour to the center of the contour. If the
                               distance is larger, the segment will be ignored.
                               Defaults to 0.1.
        run (int, optional): A higher number means a higher accuracy. Defaults to 5.
        ellipse (bool, optional): If an ellipse should be fitted to the contours.
                                  Defaults to True.
        color (str, optional): The color of the fitted ellipse. Defaults to "red".
        counts (int, optional): Defines the minimum size of a segment of the
                                contour. If the size is smaller, the segment
                                will be ignored. Defaults to 5.
        xlabel (str, optional): Label for the x-axis. Defaults to "DTR".
        ylabel (str, optional): Label for the y-axis. Defaults to "TFR".
        plot (bool, optional): If the contour plot should be saved.
                               Defaults to True.
        pixel (int, optional): The pixel, which recorded the data. Defaults to 0.
        path (str, optional): The directory, where the plot should be saved.
                              Defaults to ".".

    Returns:
        pyplot.contours, float, int: The contours of the density plot and the
                                     used threshold and iteration depth.
    """

    if xdata.shape[0] == 0:
        return None, thr, run

    # Generate the mesh.
    x, y, z = get_density_data(xdata, ydata, limit=True)

    # Initialize the figure.
    gl.set_matplotlib_rc()
    fig = plt.figure()
    ax = fig.gca()

    # Get the contours.
    ax.contourf(x, y, z, cmap=my_cmap, levels=levels)
    cset = ax.contour(x, y, z, colors='k', levels=levels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Fit ellipses to the contours.
    if ellipse:
        for i in range(levels + 1):
            plot_ellipse(get_max_area(
                i, cset, thr=thr, run=run, counts=counts), color=color)

    # Save the density plot.
    if plot:
        file_name = "Channel" + str(pixel) + "_" + xlabel + "_" + \
                    ylabel + "_matplot.png"
        plt.title("Channel " + str(pixel))
        plt.savefig(os.path.join(path, file_name))
        #plt.show()
    plt.clf()
    return cset, thr, run


# ****** The following is taken from ************************
# http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
# ************** Begin **************************************


def fitEllipse(x, y):
    """ Fits an ellipse to the data points.

    Args:
        x (np.array): X-values.
        y (np.array): Y-values.

    Returns:
        numpy.ndarray: Ellipse parameters.
    """

    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1
    E, V = np.linalg.eig(np.dot(np.linalg.inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:, n]
    return a


def ellipse_center(a):
    """ Returns the center coordinates of an ellipse.
    [summary]

    Args:
        a (numpy.ndarray): Ellipse parameters.

    Returns:
        np.array: 1d array consiting of the center coordinates.
    """

    b, c, d, f, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[0]
    num = b * b - a * c
    x0 = (c * d - b * f) / num
    y0 = (a * f - b * d) / num
    return np.array([x0, y0])


def ellipse_angle_of_rotation(a):
    """ Returns the rotation of the ellipse.

    Args:
        a (numpy.ndarray): Ellipse parameters.

    Returns:
        float: The rotation angle in radians.
    """

    b, c, a = a[1] / 2, a[2], a[0]
    return 0.5 * np.arctan(2 * b / (a - c))


def ellipse_axis_length(a):
    """ Determines the major and minor axes of an ellipse.
    Args:
        a (numpy.ndarray): Ellipse parameters.

    Returns:
        numpy.array: 1d array consiting of the lengths of the major and minor axes.
    """

    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]

    b2 = b * b
    ac = a * c
    ca_minus = a - c
    ca_plus = a + c

    b2ac = b2 - ac
    root = ca_minus * np.sqrt(1. + 4. * b2 / (ca_minus * ca_minus))

    up = 2. * (a * f * f + c * d * d + g * b2 - 2 * b * d * f - ac * g)
    down1 = -1. * b2ac * (root + ca_plus)
    down2 = b2ac * (root - ca_plus)

    if down1 * down2 <= 0:
        return np.array([0., 0.])

    res1 = np.sqrt(up / down1)
    res2 = np.sqrt(up / down2)
    return np.array([res1, res2])


# ************** End ******************************************


def get_ellipse_points(center, a, b, phi):
    """ Get points located at the ellipse.

    Args:
        center (np.array): The center coordinates of the ellipse
        a (float): The length of the major axis of the ellipse.
        b (float): The length of the minor axis of the ellipse.
        phi (float): The rotation angle in radians of the ellipse.

    Returns:
        np.array, np.array: The x and y coordinates of the points.
    """

    # Show the full ellipse: 2 pi
    arc = 2
    # Generate different angles.
    R = np.arange(0, arc * np.pi, 0.01)

    # Calculate the ellipse coordinates for differnet angles.
    x = center[0] + a * np.cos(R) * np.cos(phi) - b * np.sin(R) * np.sin(phi)
    y = center[1] + a * np.cos(R) * np.sin(phi) + b * np.sin(R) * np.cos(phi)

    return x, y



def get_Ellipse_Parameter(contour, plot=False):
    """ Calculates the parameters of the ellipse fitted to the contour.

    Args:
        contour (numpy.ndarray): 2d array of points defining the contour.
        plot (bool, optional): If the ellipse should be shown. Defaults to False.

    Returns:
        numpy.ndarray, float, float, float: 1d array containing the center
                                            coordinates, the major axis, the
                                            minor axis and the rotation angle.
    """

    if contour is None:
        return np.array([0, 0]), 0, 0, 0

    # Separate the coordinates of the points defining the contour.
    points_x = contour[:, 0]
    points_y = contour[:, 1]

    if len(points_x) == 0:
        # There are no points.
        return np.array([0, 0]), 0, 0, 0

    if plot:
        # Show the points.
        plt.plot(points_x, points_y, '.')

    # Fit an ellipse to the points and get the ellipse parameters.
    ell = fitEllipse(points_x, points_y)
    center = ellipse_center(ell)
    phi = ellipse_angle_of_rotation(ell)
    a, b = ellipse_axis_length(ell)

    if plot:
        # Show the ellipse.
        x, y = get_ellipse_points(center, a, b, phi)

        # Plot the ellipse.
        plt.plot(x, y, color='red')
        plt.show()
        plt.clf()

    return center, a, b, phi


def is_in_ellipse(x, y, center, a, b, phi):
    """ Checks if given points coordinates are inside the ellipse defined by
        the given paramters.

    Args:
        x (np.array): X-coordinates of points.
        y (np.array): Y-coordinates of points.
        center (np.array): The center coordinates of the ellipse
        a (float): The length of the major axis of the ellipse.
        b (float): The length of the minor axis of the ellipse.
        phi (float): The rotation angle in radians of the ellipse.

    Returns:
        np.array: An array of booleans saying if the points are inside the ellipse.
    """

    sin_p = np.sin(phi)
    cos_p = np.cos(phi)

    dx = x - center[0]
    dy = y - center[1]

    u = (cos_p * dx + sin_p * dy)**2 / a**2
    v = (cos_p * dy - sin_p * dx)**2 / b**2

    return u + v <= 1.


def plot_ellipse(contour, color):
    """ Plot the ellipse fitted to the contour.

    Args:
        contour (numpy.ndarray): 2d array of points defining the contour.
        color (matplotlib.color): The color of the ellipse.
    """

    if len(contour) == 0:
        return

    # Get the ellipse parameters.
    center, a, b, phi = get_Ellipse_Parameter(contour)

    # Generate coordinates of the ellipse data points.
    x, y = get_ellipse_points(center, a, b, phi)

    # Plot the ellipse.
    plt.plot(x, y, color=color, label="Fit", linewidth=3)


def get_all_data(path, days, pixels, new=False):
    """ Load all data of the given pixels of given data sets..

    Args:
        path (string): The path to the parent directory of the data sets.
        days (list like): List of names of data sets.
        pixels (list like): List of pixel numbers.
        new (bool, optional): True if the pulse shape parameters should be
                              recalculated, or False if the precalculated data
                              should be loaded. Defaults to False.

    Returns:
        pandas.DataFrame, numpy.array, numpy.array: Data frame consisting the
                                                    pulse shape parameters of
                                                    each signal, a list of names
                                                    of data sets to which the
                                                    signals corresponds to and
                                                    the number of the pixels,
                                                    which recorded the signals.
    """

    # TODO: The list of pixel numbers could be removed, since this information
    # is also located in the data frame.
    # Initialize the output.
    data = None
    list_days = None
    list_pxs = None

    # Iterate over all data sets.
    for day in days:
        print(day)

        # Load/Generate the pulse shape parameters for each pixel.
        for pixel in pixels:
            _ = pdg.PixelDay(os.path.join(path, day), pixel, new=new)

            # Append the data to the output arrays.
            if data is None:
                data = _.Data.copy()
                list_days = np.array([day] * _.Length)
                list_pxs = np.array([pixel] * _.Length)
            else:
                data = data.append(_.Data, ignore_index=True)
                list_days = np.append(list_days, np.array([day] * _.Length))
                list_pxs = np.append(list_pxs, np.array([pixel] * _.Length))

    return data, list_days, list_pxs


def plot_data_ellipse(
        xdata, ydata, center, a, b, phi, xlabel="DTR", ylabel="TFR",
        pixel=0, path="."):
    """ Save a figure of the density plot and the ellipse.

    Args:
        xdata (np.array): Data x-values.
        ydata (np.array): Data y-values.
        center (np.array): The center coordinates of the ellipse
        a (float): The length of the major axis of the ellipse.
        b (float): The length of the minor axis of the ellipse.
        phi (float): The rotation angle in radians of the ellipse.
        xlabel (str, optional): Label for the x-axis. Defaults to "DTR".
        ylabel (str, optional): Label for the y-axis. Defaults to "TFR".
        pixel (int, optional): The pixel, which recorded the data. Defaults to 0.
        path (str, optional): The directory, where the plot should be saved.
                              Defaults to ".".
    """

    # Generate the mesh.
    x, y, z = get_density_data(xdata, ydata, limit=False)

    # Initialize the figure.
    gl.set_matplotlib_rc()
    fig = plt.figure()
    ax = fig.gca()

    # Get the contours.
    ax.contourf(x, y, z, cmap=my_cmap, levels=4)
    cset = ax.contour(x, y, z, colors='k', levels=4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Plot the ellipse
    xx, yy = get_ellipse_points(center, a, b, phi)
    plt.plot(xx, yy, color='red', label="Fitted Ellipse", linewidth=3)

    # Save the plot.
    file_name = "Channel" + str(pixel) + "_" + xlabel + "_" + \
            ylabel + "_ellipse.png"
    plt.savefig(os.path.join(path, file_name))

    #plt.show()
    plt.clf()


def smoothstep(x, width, dx, a):
    """ Calculates a smooth step function.

    Args:
        x (numpy.array): The x-values.
        width (float): The maximum x range.
        dx (float): The position of the step.
        a (float): The step height.

    Returns:
        numpy.array: The calculated y-values for each x.
    """

    # ******************** Taken from ************************
    # https://stackoverflow.com/questions/45165452/how-to-implement-a-smooth-clamp-function-in-python
    # ********************************************************
    _x = np.clip((x - dx) / width, 0, 1)
    result = 0
    N = 5
    for n in range(0, N + 1):
        result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-_x) ** n
    result *= _x ** (N + 1)
    return a * result


def get_97(data, c, a, b, phi, func_x=DTR, func_y=TFR, pca=False,
        path=".", xlabel="", ylabel="", c0=0.8, c1=0.4):
    """ Find an ellipse, which described the data best.

    Args:
        data (pandas.DataFrame): Data frame of data.
        c (numpy.array): The center coordinates of the ellipse
        a (float): The length of the major axis of the ellipse.
        b (float): The length of the minor axis of the ellipse.
        phi (float): The rotation angle in radians of the ellipse.
        func_x (function, optional): A function generating a type of pulse
                                     shape parameter. Defaults to DTR.
        func_y (function, optional): A function generating a type of pulse
                                     shape parameter. Defaults to TFR.
        pca (bool, optional): False if the fraction of the two pulse shape
                              parameters should be returned. True if a principle
                              component axis should be determined. Defaults to False.
        path (str, optional): The directory, where the plot should be saved.
                              Defaults to ".".
        xlabel (str, optional): Label for the x-axis. Defaults to "".
        ylabel (str, optional): Label for the y-axis. Defaults to "".
        c0 (float, optional): Start scale factor. Defaults to 0.8.
        c1 (float, optional): Start scale factor. Defaults to 0.4.

    Returns:
        float, (int): The scale of the best ellipse (and 1, TODO to be removed)
    """

    # Define the pulse shape paramters, which scale with c0
    C0_FUNCS = [DTR, DFR, DIR]
    # AVERAGE = np.ones(5)

    # Apply the start scale factors to the axes.
    if func_x in C0_FUNCS:
        sa = c0
    elif (func_x == TFR):
        sa = 1.8 * c1
    else:
        sa = c1

    if func_y in C0_FUNCS:
        sb = c0
    elif (func_y == TFR):
        sb = 1.7 * c1
    else:
        sb = c1

    # Calculate the maximal ranges.
    MAX_A = np.abs(sa * np.cos(phi)) + np.abs(sb * np.sin(phi))
    MAX_B = np.abs(sa * np.sin(phi)) + np.abs(sb * np.cos(phi))

    # Set the start scale.
    if (MAX_A / a) > (MAX_B / b):
        SCALE = MAX_A / a
    else:
        SCALE = MAX_B / b

    #print(type(a))
    #print(b)

    # Get the pulse shape parameters.
    XDATA = func_x(data, pca=pca)
    YDATA = func_y(data, pca=pca)

    # Apply the energy cut to identify events whose reduced chi2 should be used.
    CHI_CUT = (data[gl.COLUMN_FILTER_AMP] < 1000.) & \
            (data[gl.COLUMN_FILTER_AMP] > 250.)
    CHIX = XDATA[CHI_CUT]
    CHIY = YDATA[CHI_CUT]
    CHI_DATA = data[gl.COLUMN_TEMPLATE_CHI][CHI_CUT]

    # The max scale will be reduced in the following in each iteration.
    # The values are motivated from observations of varying parameters.

    # Define the maximal reduction in %.
    MAXP = smoothstep(SCALE, 36.6848, -9.5503, 1) * (
        86.2136 - 0.34837 * SCALE + 0.00621 * SCALE**2)
    # Define the minimum reduction in %.
    PSHIFT = smoothstep(SCALE, 19.0933, 14.1261, 74.)

    # if SCALE > 20:
    #    PSHIFT = 65
    print(SCALE)

    # Generate the step length of the iterations.
    P_LENGTH = 100
    P_RANGE = range(P_LENGTH)
    PERCENTAGES = np.linspace(PSHIFT, MAXP, P_LENGTH)

    # P5_LEN = P_LENGTH - 10
    # P5_RANGE = range(P5_LEN)

    # The rms values, which will be calculated rms later, will be smoothed by
    # using an averaging kernel of length 11. Thus the percentages need to be
    # shrunk.
    KERNEL = np.exp(-(np.arange(11) - 5)**2 / 3**2)
    OUT_PER = PERCENTAGES[5:-5]
    # Translate the reduction to a scale.
    X_OUT = 0.01 * (-OUT_PER + 100.) * SCALE
    # Normalize the scales to -1.
    FIT_PER = -OUT_PER / OUT_PER[-1]

    # Generate a histogram of the chi2 values and calculate the mean value.
    VALUES, EDGES = np.histogram(CHI_DATA, range=(0, 3), bins=800)
    EDGES = EDGES[:-1]
    MAX_V = VALUES.max()
    MEAN_CHI = EDGES[VALUES == MAX_V][0]
    DELTA_CHI = 1. - MEAN_CHI

    # Define the left and right flanks of the chi2 distribution.
    LEFT = EDGES < MEAN_CHI
    RIGHT = EDGES > MEAN_CHI

    EDGES_L = EDGES[LEFT]
    EDGES_R = EDGES[RIGHT]
    VALUES_L = VALUES[LEFT]
    VALUES_R = VALUES[RIGHT]

    # Different ranges of the chi2 spectrum will be fitted by the chi2 distribution.
    # Get the boundaries of these ranges.
    # The number of ranges.
    C_LENGTH = 12
    # The percentage boundaries.
    MAX_FRACTIONS = np.linspace(0.1, 0.95, C_LENGTH)
    MIN_FRACTIONS = np.linspace(0.1, 0.95, C_LENGTH)
    # Initialize the arrays containing the chi2 values corresponding to the boundaries.
    MAX_CHIS = np.zeros(C_LENGTH)
    MIN_CHIS = np.zeros(C_LENGTH)
    # Calculate the chi2 values.
    for i in range(C_LENGTH):
        MAX_CHIS[i] = EDGES_R[VALUES_R < MAX_FRACTIONS[i] * MAX_V][0]
        MIN_CHIS[i] = EDGES_L[VALUES_L < MIN_FRACTIONS[i] * MAX_V][-1]

    # Generate chi2 distributions of events inside ellipses with differnet scales.
    # Initialize the dictionary containing the chi2 spectra.
    data_dict = {}
    # It need to be checked if there are any counts in the the ellipse.
    use_dict = np.zeros(P_LENGTH, dtype=np.bool)

    bins = int(2000 * (MAX_CHIS[0] - MIN_CHIS[0]))
    # Iterate over all ellipse scales.
    for i, p in enumerate(PERCENTAGES):
        # Define the scale.
        _s = SCALE * (1. - p / 100)
        # Get events, which are inside the ellipse.
        bool_arr = is_in_ellipse(CHIX, CHIY, c, _s * a, _s * b, phi)
        # Fill the dictionary with the chi2 distribution.
        data_dict[i], tmp_edges = np.histogram(
            CHI_DATA[bool_arr], range=(MIN_CHIS[0], MAX_CHIS[0]), bins=bins)

        # Check if there are any counts.
        max_value = data_dict[i].max()
        if (max_value > 0):
            use_dict[i] = True
            # Normalize the spectrum to 1.
            data_dict[i] = data_dict[i] / max_value

    # Get the chi2 axis.
    CHI_EDGES = tmp_edges[:-1]
    #print(CHI_DATA)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Define fit conditions for the chi2 fit.
        P0 = (350, 350, 2e-2, 70)
        BOUNDS = ((100, 100, 1e-4, 10), (550, 550, 1, 200))

        # Define fit conditions for the beta fit.
        MIN_DX = (101.725 - 0.316 * SCALE) * \
                smoothstep(SCALE, 32.743, 7.195, 1) + 4.280
        MAX_S = 2e5

        # Define the fit functions.
        def chi2_fit(chi2, df, scale, l, a):
            rv = ncx2(df, l)
            return a * rv.pdf((DELTA_CHI + chi2) * scale)

        def beta_fit(x, a, b, s, dx):
            return s * beta.pdf(x + dx / 200., a / 50., b) / 1e4

        # Initialize the list of rms of the chi2 distribution to the best fit.
        rms_list = np.zeros(P_LENGTH)
        # Initialize the best fit parameters.
        beta_rms = 1e10
        best_amp = 0

        # Iterate over all chi2 ranges.
        for min_chi in tqdm(MIN_CHIS):
            for max_chi in MAX_CHIS:
                # Initialize the last calculated rms.
                last_rms = 0.1

                # Get the chi2 range.
                cut = (CHI_EDGES <= max_chi) & (CHI_EDGES >= min_chi)
                edges = CHI_EDGES[cut]

                # Fit a chi2 distribution to the chi2 spectra of events inside
                # ellipses with different scales.
                for i in P_RANGE:
                    if use_dict[i]:
                        # Get the chi2 spectra, if there are any counts.
                        values = data_dict[i][cut]
                        try:
                            # Fit the spectra.
                            popt, _ = curve_fit(
                                chi2_fit, edges, values, p0=P0, bounds=BOUNDS)
                            # Calculate the rms.
                            tmp_rms = (chi2_fit(edges, *popt) - values)**2
                            last_rms = tmp_rms.sum() / len(edges)
                        except RuntimeError:
                            # Unable to fit the chi2 distribution to the spectra.
                            pass
                    # Assign the rms value to the ellipse scale. Use the rms
                    # calculated in the previous iteration, if the fit fails.
                    rms_list[i] = last_rms


                # Smooth the rms distribution.
                out_rms = np.convolve(rms_list, KERNEL, mode="full")[10:-10]
                # plt.plot(X_OUT, out_rms)
                # plt.show()
                # plt.clf()

                # Get the limes rms for large ellipses. It should not change,
                # if the scales are even larger. Only rms values smaller than
                # the limes will be fitted by a beta distribution.
                out_mean = out_rms[:10].mean() + out_rms[:10].std()
                fit_range = out_rms <= out_mean

                # Determine some start values for the beta fit.
                start_dx = 4. - 200 * FIT_PER[out_rms == out_rms.min()][0]
                if start_dx > 200:
                    start_dx = 200
                elif start_dx < 2 * MIN_DX:
                    start_dx = 2 * MIN_DX

                start_s = 3e6 * (out_mean - out_rms.min())
                if start_s < 1:
                    start_s = 1.
                elif start_s > MAX_S:
                    start_s = MAX_S

                # Define the fit conditions.
                if out_rms[0] < out_rms[-1]:
                    p0_beta = [75, 90., start_s, start_dx]
                    bounds = ((50., 5., 1., MIN_DX * 2), (400., 300., MAX_S, 200.))
                else:
                    p0_beta = [75., 90., start_s, 200.]
                    bounds = ((50., 5., 1., 200.), (400., 300., MAX_S, 220.))


                try:
                    # Get the x- and y-axes for the beta fit.
                    x_fit = FIT_PER[fit_range]
                    y_fit = 1e4 * (out_mean - out_rms[fit_range])

                    # Fit a beta distribution to the data.
                    popt, _ = curve_fit(
                        beta_fit, x_fit, y_fit, p0=p0_beta, bounds=bounds)

                    # plt.plot(X_OUT, out_rms, "o")
                    # plt.plot(X_OUT, out_mean - beta_fit(FIT_PER, *popt) / 1e4)
                    # plt.show()
                    # plt.clf()
                    # print(popt)

                    # Determine the rms value.
                    _beta_rms = (beta_fit(x_fit, *popt) - y_fit)**2
                    _beta_rms = _beta_rms.sum() / len(x_fit)

                    # Get the ellipse, which describes the data best.
                    # Meaning the lowest beta rms value and the highest peak to
                    # peak ratio.
                    if (beta_rms > _beta_rms) and (best_amp < start_s):
                        beta_rms = _beta_rms
                        best_amp = start_s
                        best_min = MIN_FRACTIONS[MIN_CHIS == min_chi]
                        best_max = MAX_FRACTIONS[MAX_CHIS == max_chi]
                        best_popt = np.append(popt, out_mean)
                        best_rms = out_rms.copy()
                except (RuntimeError, ValueError):
                    continue

    print(best_min)
    print(best_max)

    # Define the axis labels of the figure.
    XLABEL = tex_to_uni("Ellipse scale")
    YLABEL = "Sum of squared residuals"

    # Define the shape of the shown beta distribution. It varies from the fitted
    # one due to runtime optimizations.
    def out_beta(x, a, b, s, dx, c):
        return c - s * beta.pdf(x + dx / 200., a / 50., b) / 1e8

    # Generate the figure.
    plt.plot(X_OUT, best_rms, label="Data", color="blue")
    plt.plot(X_OUT, out_beta(FIT_PER, *best_popt), color="orange", label="Fit")
    plt.xlabel(XLABEL)
    plt.ylabel(YLABEL)
    plt.legend()

    # Generate the file name of the figure.
    APPENDIX = xlabel + "_" + ylabel + "_"

    # Save the figure.
    plt.savefig(os.path.join(path, APPENDIX + "Scale.png"))
    plt.clf()

    # Calculate the position of the minimum of the the beta distribution
    # described by the best fit parameters.
    a = best_popt[0] / 50.
    b = best_popt[1]
    dx = best_popt[3] / 200.
    mean = 1. / (1 + ((b - 1) / (a - 1)))
    var = a * b / ((a + b + 1) * (a + b)**2)

    MEAN = dx * (OUT_PER[-1] - OUT_PER[-1] * mean)
    VAR =  np.sqrt(dx**2 * OUT_PER[-1]**2 * var)

    # Translate the position to the best ellipse scale.
    if PSHIFT > 0:
        out_s = SCALE * (1. - 0.01 * (MEAN - VAR))
    else:
        out_s = SCALE * (1. - 0.01 * MEAN)

    #while out_s < 1:
    #    out_s = SCALE * (1. - 0.01 * (MEAN - s_count * VAR))

    print(out_s)
    # TODO: Remove the second return value.
    return out_s, 1


def get_parameters(data, plot=False, pixel=0, path=".", m_exists=False, pca=False):
    """ Determines the best ellipses describing the data in different parameter spaces.

    Args:
        data (pandas.DataFrame): Data frame of data.
        plot (bool, optional): If the contour plot should be saved. Defaults to False.
        pixel (int, optional): The pixel, which recorded the data. Defaults to 0.
        path (str, optional): The directory, where the plot should be saved.
                              Defaults to ".".
        m_exists (bool, optional): If the Ho-163 M1 line exists. Defaults to False.
        pca (bool, optional): False if the fraction of the two pulse shape
                              parameters should be returned. True if a principle
                              component axis should be determined. Defaults to False.

    Returns:
        list, list, list, list, list, list: Lists of centers, of major and minor axes,
                                      of rotation angles, of scales and ratios.
                                      TODO: Latter should be removed.
    """

    # Initialize the list of ellipse parameters. The ellipses are calculated in
    # different parameter spaces.
    center_list = []
    a_list = []
    b_list = []
    scale_list = []
    phi_list = []

    # TODO: Remove.
    ratio_list = []

    # Iterate over all parameter spaces.
    for i, func_x in enumerate(X_FUNCS):
        xlabel = X_LABELS[i]
        func_y = Y_FUNCS[i]
        ylabel = Y_LABELS[i]

        # Get the contours of the density plot of the current parameter space.
        contours, thr, run = get_contours(
            func_x(data, pca=pca), func_y(data, pca=pca), plot=plot,
            levels=4, thr=0.06, run=5, counts=15,
            xlabel=xlabel, ylabel=ylabel, pixel=pixel, path=path)

        # print(contours)

        # Fit an ellipse to the contour.
        center, a, b, phi = get_Ellipse_Parameter(
            get_max_area(3, contours, thr=thr, run=run, counts=15), plot=plot)

        # If the center of the ellipse is not close to (1, 1), use the innerest
        # contour.
        if (np.sqrt((center[0] - 1.)**2 + (center[1] - 1.)**2) > 0.025) or \
                (a * b == 0):
            center, a, b, phi = get_Ellipse_Parameter(
                get_max_area(4, contours, thr=thr, run=run, counts=15),
                plot=plot)

        # Add the ellipse parameters to the lists.
        center_list.append(center)
        a_list.append(a)
        b_list.append(b)
        phi_list.append(phi)
        # print("a: " + str(a) + ", b: " + str(b))
        # print(center)
        # print(phi)


        # Determine the best scale of the ellipse, which describes the data best.
        # TODO: Remove ratio.
        scale, ratio = get_97(
            data, center, a, b, phi, func_x=func_x, func_y=func_y,
            m_exists=m_exists, pca=pca)
        scale_list.append(scale)
        ratio_list.append(ratio)

        # Save the figure.
        if plot and (ratio != 0):
            plot_data_ellipse(func_x(data, pca=pca), func_y(data, pca=pca),
                center, a * scale, b * scale , phi, xlabel=xlabel, ylabel=ylabel)

    # TODO: Instead of returning multiple list, return a class.

    return center_list, a_list, b_list, phi_list, scale_list, ratio_list


def get_cut_array(data, center_list, a_list, b_list, phi_list, scale_list,
        caldata=None, pca=False):
    """ Calculates which data points are inside all ellipses defined by the
        parameters stored in the lists.

    Args:
        data (pandas.DataFrame): Data frame of data.
        center_list (list): List of ellipse centers.
        a_list (list): List of major axes.
        b_list (list): List of minor axes.
        phi_list (list): List of rotating angles.
        scale_list (list): List of ellipse scales.
        caldata (pandas.DataFrame, optional): Data used for calibration.
                                              Defaults to None.
        pca (bool, optional): False if the fraction of the two pulse shape
                              parameters should be returned. True if a principle
                              component axis should be determined. Defaults to False.

    Returns:
        numpy.array: 1d array of booleans representing if the data points are
                     inside the ellipses.
    """

    # Initialize the output.
    in_ellipse = np.ones(data.shape[0])

    # Iterate over all parameter spaces.
    for i in range(3): # len(X_FUNCS)):
        func_x = X_FUNCS[i]
        func_y = Y_FUNCS[i]

        # Get the current ellipse parameters.
        the_center = center_list[i]
        the_a = a_list[i] * scale_list[i]
        the_b = b_list[i] * scale_list[i]
        the_phi = phi_list[i]

        # Check if the data points are inside the ellipse.
        in_ellipse = in_ellipse & is_in_ellipse(
            func_x(data, caldata=caldata, pca=pca),
            func_y(data, caldata=caldata, pca=pca),
            the_center, the_a, the_b, the_phi)

    return in_ellipse


def get_cut_array_mod(
        data, center_list, a_list, b_list, phi_list, scale_list, pca=False):
    """ Calculates which data points are at least inside one ellipses defined
        by the parameters stored in the lists.

    Args:
        data (pandas.DataFrame): Data frame of data.
        center_list (list): List of ellipse centers.
        a_list (list): List of major axes.
        b_list (list): List of minor axes.
        phi_list (list): List of rotating angles.
        scale_list (list): List of ellipse scales.
        pca (bool, optional): False if the fraction of the two pulse shape
                              parameters should be returned. True if a principle
                              component axis should be determined. Defaults to False.

    Returns:
        numpy.array: 1d array of booleans representing if the data points are
                     inside the ellipses.
    """

    # Initialize the output.
    in_ellipse = np.zeros(data.shape[0])

    # Iterate over all parameter spaces.
    for i, func_x in enumerate(X_FUNCS):
        func_y = Y_FUNCS[i]

        # Get the current ellipse parameters.
        the_center = center_list[i]
        the_a = a_list[i] * scale_list[i]
        the_b = b_list[i] * scale_list[i]
        the_phi = phi_list[i]

        # Check if the data points are inside the ellipse.
        in_ellipse = in_ellipse | is_in_ellipse(
            func_x(data, pca=pca), func_y(data, pca=pca),
            the_center, the_a, the_b, the_phi)

    return in_ellipse


class EllipseCalPixel:
    """ For a given pixel, identify the pixel, which should be used as
        calibration source.

        Attributes:
            Channel (uint8): The ADC channel.
            Path (string): The path to the directory containing ADC channel directories.
            Pixel (uint8): The pixel number.
            Polarity (string): The polarity of the pulse. Either 'NEGP' or 'POSP'.
    """

    def __init__(self, path, number, polarity=None):
        """
        Args:
            path (string): Path to the directory containing ADC channel directories.
            number (int): Either a pixel number or the ADC channel.
            polarity (string, optional): The pixel polarity. Either 'NEGP' or
                                         'POSP'. Defaults to None.
        """

        # Initialize the attributes with the given pixel.
        __pulse = pg.RandomPulse(path, number, polarity=polarity)
        self.Path = __pulse.Path
        self.Channel = __pulse.Channel
        self.Polarity = __pulse.Polarity
        self.Pixel = __pulse.Pixel

        del __pulse


        # Check if the current pixel is included in data sets of the two lists
        # from above. There are some special cases. It is hard to identify
        # Ho-163 lines in coincidental events. In those cases, use non-coincidence
        # data of the same pixel (if existing).
        # TODO: This is identicall to the part in __init__ of Filter and
        # PixelDay in pixel_day_generator.
        if "only_coincidences_corr" in path:
            # Generate the path to the other pixel.
            tmp_path_0 = path.split("only_coincidences_corr")[0]
            tmp_path_1 = path.split("only_coincidences_corr")[1]
            tmp_path = tmp_path_0 + "asymmetric_channels" + tmp_path_1

            tmp_path_0 = tmp_path.split("Run24-Coincidences")[0]
            tmp_path_1 = tmp_path.split("Run24-Coincidences")[1]
            tmp_path = tmp_path_0 + "Run24-Asymmetric" + tmp_path_1

            # Check if the same pixel exists in the other path.
            # If yes, load this pixel.
            if os.path.exists(os.path.join(
                    tmp_path, "ADC" + str(self.Channel), self.Polarity)):

                _ = pg.RandomPulse(tmp_path, number, polarity=polarity)
                temp_cal = EllipseCalPixel(
                    tmp_path, number, polarity=polarity)
                self.Path = temp_cal.Path
                self.Channel = temp_cal.Channel
                self.Polarity = temp_cal.Polarity
                self.Pixel = temp_cal.Pixel
                return

        # Get the path to the file consiting the calibration lines.
        # Generate the file, if it does not exist.
        pathToDirectory = os.path.join(
            self.Path, gl.ADC + str(self.Channel), self.Polarity)
        pathToIndex = os.path.join(pathToDirectory, gl.FILE_CALIBRATION_INDEX)
        if not os.path.exists(pathToIndex):
            _ = cg.CalibrationEvents(
                self.Path, self.Channel, polarity=self.Polarity)
            del _

        # Check, if there are any peaks in the data set.
        if pd.read_csv(pathToIndex)[gl.COLUMN_CAL_INDEX][0] == 0:
            # If not, try the partner pixel of the same channel.
            tmp_pol = gl.NEGP
            if self.Polarity == gl.NEGP:
                tmp_pol = gl.POSP

            pathToDirectory = os.path.join(
                self.Path, gl.ADC + str(self.Channel), tmp_pol)
            pathToIndex = os.path.join(
                pathToDirectory, gl.FILE_CALIBRATION_INDEX)
            if not os.path.exists(pathToIndex):
                _ = cg.CalibrationEvents(
                    self.Path, self.Channel, polarity=tmp_pol)
                del _

            if pd.read_csv(pathToIndex)[gl.COLUMN_CAL_INDEX][0] == 0:
                # There are also no lines detected in the data set of the partner
                # pixel. Use the default pixel.
                __pulse = pg.RandomPulse(gl.DEFAULT_PATH, gl.DEFAULT_PIXEL)
            else:
                __pulse = pg.RandomPulse(
                    self.Path, self.Channel, polarity=tmp_pol)

            # Assign the attributes.
            self.Path = __pulse.Path
            self.Channel = __pulse.Channel
            self.Polarity = __pulse.Polarity
            self.Pixel = __pulse.Pixel

            del __pulse


def get_ellipse_cut(pixelday, new=False, plot=False, rec=0):
    """ Calculates the ellipses describing the data best and determines, which
        data points are inside all ellipses.

    Args:
        pixelday (PixelDay): The pixel data.
        new (bool, optional): True if the ellipses parameters should be
                              recalculated, or False if the precalculated data
                              should be loaded. Defaults to False.
        plot (bool, optional): If the contour plot should be saved. Defaults to False.
        rec (int, optional): Level of recursions. Defaults to 0.
                             Should not be changed.

    Returns:
        numpy.array: 1d array of booleans representing if the data points are
                     inside the ellipses.
    """

    # The range of the pulse shape parameters = np.abs(parameter - 1)
    WINDOW_CUT = 0.4
    # Get the calibration pixel for this data set.
    cal_pix = EllipseCalPixel(
        pixelday.Path, pixelday.Channel, polarity=pixelday.Polarity)

    # Directory where the ellipses parameters are/will be stored.
    path_to_data = os.path.join(
        cal_pix.Path, gl.ADC + str(cal_pix.Channel), cal_pix.Polarity)

    # Load the calibration lines.
    calibration_lines = pd.read_csv(os.path.join(
        path_to_data, gl.FILE_CALIBRATION_LINES))

    # Check if the Ho-163 M1 line is found.
    found_m = calibration_lines[gl.COLUMN_M][0] != 0

    # Do the same for the partner pixel.
    other_pol = gl.NEGP
    if pixelday.Polarity == gl.NEGP:
        other_pol = gl.POSP

    other_path =  os.path.join(
        pixelday.Path, gl.ADC + str(pixelday.Channel), other_pol)
    other_lines = pd.read_csv(os.path.join(
        other_path, gl.FILE_CALIBRATION_LINES))
    other_m = other_lines[gl.COLUMN_M][0] != 0

    if (not new) and os.path.exists(os.path.join(
            path_to_data, gl.FILE_ELLIPSE_CENTERS)):

        # Load the ellipses parameters if they exists.
        c_list = np.load(os.path.join(path_to_data, gl.FILE_ELLIPSE_CENTERS))
        a_list = np.load(os.path.join(path_to_data, gl.FILE_ELLIPSE_AS))
        b_list = np.load(os.path.join(path_to_data, gl.FILE_ELLIPSE_BS))
        p_list = np.load(os.path.join(path_to_data, gl.FILE_ELLIPSE_PHIS))
        s_list = np.load(os.path.join(path_to_data, gl.FILE_ELLIPSE_SCALES))

        # TODO: Remove anything corresponding to ratio_list.
        ratio_list = np.load(os.path.join(path_to_data, gl.FILE_ELLIPSE_RATIOS))
        # Check if the lines are dominant.
        too_much_noise = True
        for i in ratio_list:
            if i >= 0.97:
                too_much_noise = False

        if too_much_noise and (rec == 0):
            # The lines are not dominant. Use the partner pixel, if it is better.
            if not os.path.exists(os.path.join(
                    other_path, gl.FILE_ELLIPSE_CENTERS)):
                tmp = pdg.PixelDay(
                    pixelday.Path, pixelday.Channel, polarity=other_pol)
                _ = get_ellipse_cut(tmp, rec=1)

            if os.path.exists(os.path.join(
                    other_path, gl.FILE_ELLIPSE_CENTERS)):
                o_c_list = np.load(os.path.join(
                    other_path, gl.FILE_ELLIPSE_CENTERS))
                o_a_list = np.load(os.path.join(
                    other_path, gl.FILE_ELLIPSE_AS))
                o_b_list = np.load(os.path.join(
                    other_path, gl.FILE_ELLIPSE_BS))
                o_p_list = np.load(os.path.join(
                    other_path, gl.FILE_ELLIPSE_PHIS))
                o_s_list = np.load(os.path.join(
                    other_path, gl.FILE_ELLIPSE_SCALES))
                o_ratio_list = np.load(os.path.join(
                    other_path, gl.FILE_ELLIPSE_RATIOS))

                for i in o_ratio_list:
                    if i >= 0.97:
                        too_much_noise = False

                if not too_much_noise:
                    c_list = o_c_list
                    a_list = o_a_list
                    b_list = o_b_list
                    p_list = o_p_list
                    s_list = o_s_list
                    found_m = other_m

        if plot:
            # Save the contour plot.
            cal_data = pdg.PixelDay(
                cal_pix.Path, cal_pix.Channel, polarity=cal_pix.Polarity)
            IS_IN_WINDOW = (np.abs(DTR(cal_data.Data) - 1.) < WINDOW_CUT) & \
                    (np.abs(DFR(cal_data.Data) - 1.) < WINDOW_CUT) & \
                    (np.abs(DIR(cal_data.Data) - 1.) < WINDOW_CUT) & \
                    (np.abs(TFR(cal_data.Data) - 1.) < WINDOW_CUT) & \
                    (np.abs(TIR(cal_data.Data) - 1.) < WINDOW_CUT) & \
                    (np.abs(FIR(cal_data.Data) - 1.) < WINDOW_CUT) & \
                    (cal_data.Data[gl.COLUMN_TEMPLATE_CHI] < 10)
            _ = get_parameters(
                cal_data.Data[IS_IN_WINDOW], plot=plot, pixel=pixelday.Pixel,
                path=pixelday.Path, m_exists=found_m)
    else:
        # Load the calibration data.
        cal_data = pdg.PixelDay(
            cal_pix.Path, cal_pix.Channel, polarity=cal_pix.Polarity)
        # Clean the data. Use only data fullfilling some cut conditions.
        IS_IN_WINDOW = (np.abs(DTR(cal_data.Data) - 1.) < WINDOW_CUT) & \
                (np.abs(DFR(cal_data.Data) - 1.) < WINDOW_CUT) & \
                (np.abs(DIR(cal_data.Data) - 1.) < WINDOW_CUT) & \
                (np.abs(TFR(cal_data.Data) - 1.) < WINDOW_CUT) & \
                (np.abs(TIR(cal_data.Data) - 1.) < WINDOW_CUT) & \
                (np.abs(FIR(cal_data.Data) - 1.) < WINDOW_CUT) & \
                (cal_data.Data[gl.COLUMN_TEMPLATE_CHI] < 10)

        # Get the ellipse parameters.
        c_list, a_list, b_list, p_list, s_list, ratio_list = get_parameters(
            cal_data.Data[IS_IN_WINDOW], plot=plot, pixel=pixelday.Pixel,
            path=pixelday.Path, m_exists=found_m)

        # Convert the lists to array, so that they can be saved.
        c_list = np.array(c_list)
        a_list = np.array(a_list)
        b_list = np.array(b_list)
        p_list = np.array(p_list)
        s_list = np.array(s_list)

        # TODO: Remove ratio_list and anything connected to it.
        ratio_list = np.array(ratio_list)

        # Save the ellipse parameters.
        np.save(os.path.join(path_to_data, gl.FILE_ELLIPSE_CENTERS), c_list)
        np.save(os.path.join(path_to_data, gl.FILE_ELLIPSE_AS), a_list)
        np.save(os.path.join(path_to_data, gl.FILE_ELLIPSE_BS), b_list)
        np.save(os.path.join(path_to_data, gl.FILE_ELLIPSE_PHIS), p_list)
        np.save(os.path.join(path_to_data, gl.FILE_ELLIPSE_SCALES), s_list)
        np.save(os.path.join(path_to_data, gl.FILE_ELLIPSE_RATIOS), ratio_list)

        # Check, if the lines are dominant.
        too_much_noise = True
        for i in ratio_list:
            if i >= 0.97:
                too_much_noise = False

        if too_much_noise and (rec == 0):
            # The lines are not dominant. Use the partner pixel, if it is better.
            if new or (not os.path.exists(os.path.join(
                    other_path, gl.FILE_ELLIPSE_CENTERS))):
                tmp = pdg.PixelDay(
                    pixelday.Path, pixelday.Channel, polarity=other_pol)
                _ = get_ellipse_cut(tmp, new=new, rec=1, plot=plot)

            if os.path.exists(os.path.join(
                    other_path, gl.FILE_ELLIPSE_CENTERS)):
                o_c_list = np.load(os.path.join(
                    other_path, gl.FILE_ELLIPSE_CENTERS))
                o_a_list = np.load(os.path.join(
                    other_path, gl.FILE_ELLIPSE_AS))
                o_b_list = np.load(os.path.join(
                    other_path, gl.FILE_ELLIPSE_BS))
                o_p_list = np.load(os.path.join(
                    other_path, gl.FILE_ELLIPSE_PHIS))
                o_s_list = np.load(os.path.join(
                    other_path, gl.FILE_ELLIPSE_SCALES))
                o_ratio_list = np.load(os.path.join(
                    other_path, gl.FILE_ELLIPSE_RATIOS))

                for i in o_ratio_list:
                    if i >= 0.97:
                        too_much_noise = False

                if not too_much_noise:
                    c_list = o_c_list
                    a_list = o_a_list
                    b_list = o_b_list
                    p_list = o_p_list
                    s_list = o_s_list

    # Check which data points are inside the ellipses and return this.
    return get_cut_array(pixelday.Data, c_list, a_list, b_list, p_list, s_list)


def plot_pulses(plot_data, plot_days, plot_pixels, index, path_to_data):
    """ Plot a pulse labeled with the pulse shape parameters.
    Args:
        plot_data (pandas.DataFrame): Data frame of data.
        plot_days (numpy.array): List of names of data sets.
        plot_pixels (numpy.array): List of pixel numbers.
        index (int): The index of the pulse, which should be plotted.
        path_to_data (string): Path to the parent directory of the data sets.
    """

    import pulse_generator as pg
    from TexToUni import tex_to_uni

    # Get the pulse shape parameters of the pulse.
    pulse = int(plot_data[gl.COLUMN_SIGNAL_NUMBER].iloc[index])
    e_filter = np.round(plot_data[gl.COLUMN_FILTER_AMP].iloc[index])
    e_template = np.round(plot_data[gl.COLUMN_TEMPLATE_AMP].iloc[index])
    e_derivative = np.round(plot_data[gl.COLUMN_DERIVATIVE_AMP].iloc[index])
    e_integral = np.round(plot_data[gl.COLUMN_FULL_INTEGRAL].iloc[index])
    chi = np.round(plot_data[gl.COLUMN_TEMPLATE_CHI].iloc[index], decimals=2)

    # Define some figure properties.
    xlabel = tex_to_uni("Time in \mus")
    ylabel = "Voltage in mV"

    # Get the name of the data set and the pixel.
    day = plot_days[index]
    pixel = plot_pixels[index]

    day_split = day.split("_")
    day_title = "Day: " + day_split[3] + "_" + day_split[5]

    # day_count = 0
    # for d in DAYS:
    #    if d != day:
    #        day_count += 1
    #        continue
    #    if day_count < 7:
    #        day_title = "Day: 191223_" + str(day_count)
    #    elif day_count == 7:
    #        day_title = "Day: 200108_0"
    #    elif day_count == 8:
    #        day_title = "Day: 200116_0"
    #    elif day_count == 9:
    #        day_title = "Day: 200116_1"
    #    elif day_count == 10:
    #        day_title = "Day: 200120_0"
    #    else:
    #        day_title = "Day: 200122_0"
    #    break

    # Load the pulse.
    tmp_pulse = pg.Pulse(
        os.path.join(path_to_data, day), number=pixel, pulse=pulse)
    # Load the corresponding template.
    tmp_template = pdg.Filter(os.path.join(path_to_data, day), pixel)

    # A scale factor. The template and pulse don not have the same height.
    mult = e_template / 2053.

    # Define the time axis.
    xaxis = np.arange(tmp_pulse.Data.shape[0]) * 128. / 1000.
    pixel_title = "Pixel " + str(pixel) + ": Pulse " + str(pulse)
    # print(plot_data.iloc[index].shape[0])

    # Check if the pulse survives the cuts.
    if get_ellipse_cut(plot_data).iloc[index]:
        pulse_title = "Good pulse"
        pulse_color = "green"
    else:
        pulse_title = "Bad pulse"
        pulse_color = "red"

    # Define the labels.
    template_label = "Template:  " + str(e_template).split(".")[0] + " eV"
    derivative_label = "Derivative: " + str(e_derivative).split(".")[0] + " eV"
    filter_label = "Filter:         " + str(e_filter).split(".")[0] + " eV"
    integral_label = "Integral:     " + str(e_integral).split(".")[0] + " eV"
    chi_label = tex_to_uni("\chi^{2} dof^{-1} = " + str(chi))

    # Generate the plot.
    out = ptool.Curve(np.array([0., 0.1]), np.array([0.1, 0.1]),
                      title=day_title, color="black")
    out *= ptool.Curve(xaxis, tmp_pulse.Data, title=pixel_title,
                       color=pulse_color)
    out *= ptool.Curve(np.array([0., 0.1]), np.array([0.1, 0.1]),
                       title=pulse_title, color=pulse_color)
    out *= ptool.Curve(xaxis, mult * tmp_template.Data[::-1],
                       title=template_label, color="orange")
    out *= ptool.Curve(np.array([0., 0.1]), np.array([0.1, 0.1]),
                       title=filter_label, color="orange", alpha=0.)
    out *= ptool.Curve(np.array([0., 0.1]), np.array([0.1, 0.1]),
                       title=derivative_label, color="orange", alpha=0.)
    out *= ptool.Curve(np.array([0., 0.1]), np.array([0.1, 0.1]),
                       title=integral_label, color="orange", alpha=0.)
    out *= ptool.Curve(np.array([0., 0.1]), np.array([0.1, 0.1]),
                       title=chi_label, color="orange", alpha=0.)

    # Show the pulse.
    out = out.opts(xlabel=xlabel, ylabel=ylabel)
    out.plot()


def plot_sum_pulse(plot_data, plot_days, plot_pixels, path_to_data, good=True):
    """ Plot the average pulse of the given pulses.

    Args:
        plot_data (pandas.DataFrame): Data frame of data.
        plot_days (numpy.array): List of names of data sets.
        plot_pixels (numpy.array): List of pixel numbers.
        path_to_data (string): Path to the parent directory of the data sets.
        good (bool, optional): Defines if the pulses survive the cuts or not.
                               Defaults to True.
    """

    if len(plot_days) == 0:
        print("Nothing to show")
        return

    # Define some figure properties.
    xlabel = tex_to_uni("Time in \mus")
    ylabel = "Voltage in mV"

    # Do not sum over all pulses. This would need to much computing time.
    num_of_events = 30
    if plot_days.shape[0] < num_of_events:
            num_of_events = plot_days.shape[0]

    # Determine the average pulse and template.
    t_data = None
    t_template = None
    t_energy = 0.
    for i in range(num_of_events):
        day = plot_days[i]
        pixel = plot_pixels[i]
        # Load the pulse.
        pulse = int(plot_data[gl.COLUMN_SIGNAL_NUMBER].iloc[i])
        # if (day[-2:] == "_3") and ((pixel == 12) or (pixel == 12)):
        #    continue
        tmp_pulse = pg.Pulse(
            os.path.join(path_to_data, day), number=pixel, pulse=pulse)

        # Load the template.
        tmp_template = pdg.Filter(os.path.join(path_to_data, day), pixel)
        # Scale the template.
        mult = plot_data[gl.COLUMN_TEMPLATE_AMP].iloc[i] / 2053.
        t_energy += plot_data[gl.COLUMN_TEMPLATE_AMP].iloc[i]

        # Add the template and pulse to the average template and pulse.
        if t_data is None:
            t_data = tmp_pulse.Data
            t_template = tmp_template.Data[::-1] * mult
        else:
            t_data += tmp_pulse.Data
            t_template += tmp_template.Data[::-1] * mult

    # Define the time axis.
    xaxis = np.arange(tmp_pulse.Data.shape[0]) * 128. / 1000.

    p_title = "Sum good pulse: "
    if not good:
        p_title = "Sum bad pulse: "
    p_title += str(t_energy / num_of_events).split(".")[0] + " eV"
    p_color = "green"
    if not good:
        p_color = "red"

    # Create the plot.
    out = ptool.Curve(
        xaxis, t_data / num_of_events, title=p_title, color=p_color)
    out *= ptool.Curve(
        xaxis, t_template / num_of_events, title="Template", color="orange")

    out = out.opts(xlabel=xlabel, ylabel=ylabel)
    out.plot()


def generate_ellipse_parameters(path, c0=0.8, c1=0.4, ch=10., new=False):
    """ Generate the ellipses paramters for all pixels in path.

    Args:
        path (string): Path to the directory containing ADC channel directories.
        c0 (float, optional): Start scale factor. Defaults to 0.8.
        c1 (float, optional): Start scale factor. Defaults to 0.4.
        ch ([type], optional): Maximal chi2 value. Defaults to 10.
        new (bool, optional): True if the ellipses parameters should be
                              recalculated, or False if the precalculated data
                              should be loaded. Defaults to False.
    """

    # Only use events fullfilling some conditions.
    def get_window_cut(data):
        if data is None:
            return None

        out = (np.abs(DTR(data, caldata=data) - 1.) < c0) & \
                (np.abs(DFR(data, caldata=data) - 1.) < c0) & \
                (np.abs(DIR(data, caldata=data) - 1.) < c0) & \
                (np.abs(TFR(data, caldata=data) - 1.) < 2 * c1) & \
                (np.abs(TIR(data, caldata=data) - 1.) < c1) & \
                (np.abs(FIR(data, caldata=data) - 1.) < c1) & \
                (data[gl.COLUMN_TEMPLATE_CHI] < ch) & \
                (data[gl.COLUMN_TEMPLATE_CHI] > 0.4)
        return out

    # Get the name of the data set.
    day = path.split(gl.PATH_SEP)[-1]

    # Load the list of pixels.
    pathToCSV = os.path.join(path, gl.FILE_PIXEL_LIST)
    if not os.path.exists(pathToCSV):
        try:
            _ = pg.RandomPulse(path, 1)
            del _
        except pg.PixelNotFoundError:
            pass
    frame = pd.read_csv(pathToCSV)

    # Get all channels.
    unique_channels = np.unique(frame[gl.COLUMN_ADC_CHANNEL])

    # Create the ellipses parameters for all channels.
    for channel in unique_channels:

        # Define the path to the channel data.
        tmp_path = os.path.join(path, "ADC" + str(int(channel)))
        if os.path.exists(os.path.join(
                tmp_path, gl.FILE_ELLIPSE_CENTERS)) and not new:
            # Parameters are already created.
            continue

        # Load the data of both pixels of the current channel.
        data = None
        for pixel in frame[frame[gl.COLUMN_ADC_CHANNEL] == channel][
                    gl.COLUMN_PIXEL_NUMBER]:
            tmp_data = pdg.PixelDay(os.path.join(path), pixel)
            if data is None:
                data = tmp_data.Data.copy()
            else:
                data = data.append(tmp_data.Data)
            del tmp_data

        # Not all events should be used. In best case, only Ho-163 induced events.
        cut_arr = get_window_cut(data)

        if cut_arr is None:
            # No Ho-163-like events are found.
            print(channel + ": Nothing to show")
            del data
            gc.collect()
            continue

        # Initialize the list of ellipses parameters.
        center_list = []
        a_list = []
        b_list = []
        scale_list = []
        phi_list = []
        ratio_list = []

        # Save the chi2 distribution.
        plt.hist(data[gl.COLUMN_TEMPLATE_CHI], range=(0, 5), bins=100)
        plt.yscale("log")
        plt.xlabel(tex_to_uni("\Chi^{2}\,dof^{-1}"))
        plt.ylabel("Counts per 0.05")
        plt.savefig(os.path.join(tmp_path, "Chi.png"))
        plt.clf()

        # The last scale. It is only for the output.
        last_s = 0.

        # Determines the ellipses parameters for all parameter spaces.
        for i in range(len(X_FUNCS)):
            func_x = X_FUNCS[i]
            xlabel = X_LABELS[i]
            func_y = Y_FUNCS[i]
            ylabel = Y_LABELS[i]

            clear_output(wait=False)
            print(day)
            print("Channel " + str(channel) + ": Number " + str(i) + " " +
                  xlabel + "-" + ylabel)
            print("Last scale " + str(last_s))

            # Get the contours of the density plots.
            contours, thr, run = get_contours(
                func_x(data[cut_arr]),
                func_y(data[cut_arr]), plot=True,
                levels=4, thr=0.06, run=5, counts=15,
                xlabel=xlabel, ylabel=ylabel, pixel=channel, path=tmp_path)
            # Fit an ellipse to the contour.
            center, a, b, phi = get_Ellipse_Parameter(
                get_max_area(4, contours, thr=thr, run=run, counts=15),
                plot=False)

            # Add the parameters to the lists.
            center_list.append(center)
            a_list.append(a)
            b_list.append(b)
            phi_list.append(phi)

            # Determine the best ellipse scale.
            # TODO: Remove anything connected to ratio.
            scale, ratio = get_97(
                data[cut_arr], center, a, b, phi,
                func_x=func_x, func_y=func_y,
                m_exists=True, path=tmp_path, xlabel=xlabel, ylabel=ylabel)

            # Add the ellipse scale to the list.
            scale_list.append(scale)
            ratio_list.append(ratio)

            if ratio != 0:
                # Save a figure of the ellipse, if identified.
                plot_data_ellipse(
                    func_x(data[cut_arr]),
                    func_y(data[cut_arr]), center,
                    a * scale, b * scale , phi,
                    xlabel=xlabel, ylabel=ylabel, path=tmp_path, pixel=channel)

            last_s = scale

        # Save the parameters.
        np.save(os.path.join(tmp_path, gl.FILE_ELLIPSE_CENTERS), center_list)
        np.save(os.path.join(tmp_path, gl.FILE_ELLIPSE_AS), a_list)
        np.save(os.path.join(tmp_path, gl.FILE_ELLIPSE_BS), b_list)
        np.save(os.path.join(tmp_path, gl.FILE_ELLIPSE_PHIS), phi_list)
        np.save(os.path.join(tmp_path, gl.FILE_ELLIPSE_SCALES), scale_list)
        np.save(os.path.join(tmp_path, gl.FILE_ELLIPSE_RATIOS), ratio_list)
        del data
        gc.collect()


#def recalculate_scale(path, c0=0.8, c1=0.4, start=None, first=None):
def recalculate_scale(path, c0=0.05, c1=0.005, start=None, first=None):
    """ Recalculate the scales of the ellipses of the all pixels in path.

    Args:
        path (string): Path to the directory containing ADC channel directories.
        c0 (float, optional): Start scale factor. Defaults to 0.05.
        c1 (float, optional): Start scale factor. Defaults to 0.005.
        start (int, optional): First channel which should be analyzed.
                               Defaults to None.
        first (int, optional): First parameter space, which should be analyzed.
                               Defaults to None.
    """

    # Only use events fullfilling some conditions.
    def get_window_cut(data, c0, c1):
        if data is None:
            return None

        out = (np.abs(DTR(data, caldata=data) - 1.) < c0) & \
                (np.abs(DFR(data, caldata=data) - 1.) < c0) & \
                (np.abs(DIR(data, caldata=data) - 1.) < c0) & \
                (np.abs(TFR(data, caldata=data) - 1.) < 1.8 * c1) & \
                (np.abs(TIR(data, caldata=data) - 1.) < c1) & \
                (np.abs(FIR(data, caldata=data) - 1.) < c1)
        return out

    # Define the first used channel and parameter space.
    _start = start
    _first = first

    # Get the name of the data set.
    day = path.split(gl.PATH_SEP)[-1]

    # Load the list of pixels.
    pathToCSV = os.path.join(path, gl.FILE_PIXEL_LIST)
    if not os.path.exists(pathToCSV):
        try:
            _ = pg.RandomPulse(path, 1)
            del _
        except pg.PixelNotFoundError:
            pass
    frame = pd.read_csv(pathToCSV)

    # Get all channels.
    unique_channels = np.unique(frame[gl.COLUMN_ADC_CHANNEL])

    # Re-identify the best ellipses for all channels.
    for channel in unique_channels:
        if _start is not None:
            if channel < _start:
                continue
            else:
                _start = None

        # Define the path to the channel data.
        tmp_path = os.path.join(path, "ADC" + str(int(channel)))

        # Adjust the cut conditions for some channels.
        # TODO: This should not be needed, since c0 and c1 are automatically
        # identified further below.
        # if channel == 2:
        #    c0 = 0.25
        #    c1 = 0.03
        # elif channel == 6:
        #    c0 = 0.25
        #    c1 = 0.032
        # elif channel == 10:
        #    c0 = 0.075
        #    c1 = 0.0175
        # elif channel == 15:
        #    c0 = 0.2
        #    c1 = 0.035
        # else:
        #    c0 = 0.1
        #    c1 = 0.0125

        # Delete the generated figures.
        # for f in os.scandir(tmp_path):
        #    if f.name.endswith(".png"):
        #        if not f.name.startswith("Chi"):
        #            os.remove(f.path)
        # continue

        # Load the data of both pixels of the current channel.
        data = None
        for pixel in frame[frame[gl.COLUMN_ADC_CHANNEL] == channel][
                    gl.COLUMN_PIXEL_NUMBER]:
            tmp_data = pdg.PixelDay(os.path.join(path), pixel)
            if data is None:
                data = tmp_data.Data.copy()
            else:
                data = data.append(tmp_data.Data)
            del tmp_data

        # Identify the cut conditions. Change s0 and s1 (and thus c0, c1), so
        # that the number of events fullfilling the cut conditions does not change
        # much if c0 and c1 are further increased (= weaker cut).
        s0 = 1
        s1 = 1
        ds = 0.05

        # = 0.005 %
        lim = 99.995 / 100

        # TODO: Why range(3)? Need to be checked.
        for i in range(3):
            # Number of events fullfilling the current conditions.
            sum0 = get_window_cut(data, s0 * c0, s1 * c1).sum()
            # Number of events fullfilling weaker conditions.
            sum1 = get_window_cut(data, (s0 + ds) * c0, s1 * c1).sum()

            while sum0 / sum1 < lim:
                # Increase c0 until the number of events in the window does not
                # change much, if c0 increases further.
                s0 += ds
                sum0 = get_window_cut(data, s0 * c0, s1 * c1).sum()
                sum1 = get_window_cut(data, (s0 + ds) * c0, s1 * c1).sum()

            sum0 = get_window_cut(data, s0 * c0, s1 * c1).sum()
            sum1 = get_window_cut(data, s0 * c0, (s1 + ds) * c1).sum()

            while sum0 / sum1 < lim:
                # Increase c1 until the number of events in the window does not
                # change much, if c1 increases further.
                s1 += ds
                sum0 = get_window_cut(data, s0 * c0, s1 * c1).sum()
                sum1 = get_window_cut(data, s0 * c0, (s1 + ds) * c1).sum()


        # Load the ellipse parameters.
        center = np.load(os.path.join(tmp_path, gl.FILE_ELLIPSE_CENTERS))
        a = np.load(os.path.join(tmp_path, gl.FILE_ELLIPSE_AS))
        b = np.load(os.path.join(tmp_path, gl.FILE_ELLIPSE_BS))
        phi = np.load(os.path.join(tmp_path, gl.FILE_ELLIPSE_PHIS))
        scale_list = []
        ratio_list = []


        # The last scale. It is only for the output.
        last_s = 0.

        # Determines the ellipses scales for all parameter spaces.
        for i in range(3):#len(X_FUNCS)):
            if _first is not None:
                if i < _first:
                    continue
                else:
                    _first = None
            func_x = X_FUNCS[i]
            xlabel = X_LABELS[i]
            func_y = Y_FUNCS[i]
            ylabel = Y_LABELS[i]

            clear_output(wait=False)
            print(day)
            print("Channel " + str(channel) + ": Number " + str(i) + " " +
                  xlabel + "-" + ylabel)
            print("Last scale " + str(last_s))
            print(s0 * c0)
            print(s1 * c1)

            # TODO: It can occur, that there are complex parameters. This should not
            # happen. Need to be checked.
            if (a[i] == 0) or (b[i] == 0) or (type(a[i]) == np.complex128) or \
                    (type(b[i]) == np.complex128):
                scale_list.append(0)
                ratio_list.append(0)
                continue

            # Determine the best scale.
            # TODO: Remove anything connected to ratio.
            scale, ratio = get_97(
                data, center[i], a[i], b[i], phi[i],
                func_x=func_x, func_y=func_y, c0=s0 * c0, c1=s1 * c1,
                path=tmp_path, xlabel=xlabel, ylabel=ylabel)
            scale_list.append(scale)
            ratio_list.append(ratio)

            if ratio != 0:
                # Save a figure of the ellipse, if identified.
                cut_arr = get_window_cut(data, s0 * c0, s1 * c1)
                plot_data_ellipse(
                    func_x(data)[cut_arr],
                    func_y(data)[cut_arr], center[i],
                    a[i] * scale, b[i] * scale , phi[i],
                    xlabel=xlabel, ylabel=ylabel, path=tmp_path, pixel=channel)

            last_s = scale

        # Save the ellipses scales.
        np.save(os.path.join(tmp_path, gl.FILE_ELLIPSE_SCALES), scale_list)
        np.save(os.path.join(tmp_path, gl.FILE_ELLIPSE_RATIOS), ratio_list)
        del data
        gc.collect()


def load_data(path, c0=0.8, c1=0.4, ch=10.):
    """ Load the all the data in path and split it in data inside and outside
        the ellipses.

    Args:
        path (string): The path to the directory containing ADC channel directories.
        c0 (float, optional): Start scale factor. Defaults to 0.8.
        c1 (float, optional): Start scale factor. Defaults to 0.4.
        ch (float, optional): Maximal chi2 value. Defaults to 10..

    Returns:
        pandas.DataFrame, numpy.array, pandas.DataFrame, numpy.array,
        pandas.DataFrame, numpy.array, float, float:
                Data frame of all data points, an array of corresponding names
                of data sets, data frame of data points inside the ellipses,
                an array of corresponding names of data sets, data frame of
                data points outside the ellipses, an array of corresponding
                names of data sets, the number of recorded pixel-days, the
                total measurement duration in seconds.
    """

    # Initialize the outputs.
    all_data = None
    all_data_d = None

    ellipse_data = None
    ellipse_data_d = None

    non_ellipse_data = None
    non_ellipse_data_d = None

    # Path of the file containing the pixel numbers with corresponding polarities and
    # channel numbers.
    pathToCSV = os.path.join(path, gl.FILE_PIXEL_LIST)

    # Get the name of the data set.
    day = path.split(gl.PATH_SEP)[-1]

    # Initialize the time information.
    num_of_pixels = 0
    max_time = 0
    min_time = np.inf

    # Load the file containing pixel-polarity-channel information.
    # Generate it, if not existing.
    if not os.path.exists(pathToCSV):
        try:
            _ = pg.RandomPulse(path, 1)
            del _
        except pg.PixelNotFoundError:
            pass
    frame = pd.read_csv(pathToCSV)

    # Get the ADC channels.
    unique_channels = np.unique(frame[gl.COLUMN_ADC_CHANNEL])

    # Load the data of each channel/pixel pair.
    for channel in unique_channels:
        if (int(channel) == 10) and (("20200411" in path) or
                ("20200414" in path)):
            # TODO: Not analyzed yet.
            continue

        # Load the ellipse parameters.
        tmp_path = os.path.join(path, "ADC" + str(int(channel)))
        if not os.path.exists(os.path.join(path, gl.FILE_ELLIPSE_CENTERS)):
            generate_ellipse_parameters(path, c0=c0, c1=c1, ch=ch)

        center_list = np.load(os.path.join(tmp_path, gl.FILE_ELLIPSE_CENTERS))
        a_list = np.load(os.path.join(tmp_path, gl.FILE_ELLIPSE_AS))
        b_list = np.load(os.path.join(tmp_path, gl.FILE_ELLIPSE_BS))
        phi_list = np.load(os.path.join(tmp_path, gl.FILE_ELLIPSE_PHIS))
        scale_list = np.load(os.path.join(tmp_path, gl.FILE_ELLIPSE_SCALES))

        # The following channels have no/ only a few counts. Skip these.
        if ("0311" in path) and (channel == 31):
            continue
        elif ("0408" in path) and (channel == 10):
            continue
        elif ("0411" in path) and (channel == 10):
            continue
        elif ("0414" in path) and ((channel == 10) or (channel == 31)):
            continue
        elif ("0414_54" in path) and ((channel == 6) or (channel == 15)):
            continue
        elif ("0416" in path) and (channel == 31):
            continue
        elif ("0421" in path) and ((channel == 6) or (channel == 31)):
            continue
        elif ("0422_54ms_1" in path) and ((channel == 6) or (channel == 31)):
            continue
        elif ("0429" in path) and (channel == 31):
            continue
        elif ("0504" in path) and ((channel == 6) or (channel == 31)):
            continue
        elif (len(scale_list) < 3):
            continue

        # Load the data of the pixels of the current channel.
        for pixel in frame[frame[gl.COLUMN_ADC_CHANNEL] == channel][
                gl.COLUMN_PIXEL_NUMBER]:

            if (pixel not in [3, 11, 19, 30, 61]):
                continue
            #if (pixel not in [4, 12, 20, 29, 62]):
            #    continue

            num_of_pixels += 1

            # Load the data.
            _pday = pdg.PixelDay(path, pixel)
            tmp_data = _pday.Data

            # Check, which data points are inside the ellipses.
            cut_array = get_cut_array(
                tmp_data, center_list, a_list, b_list, phi_list, scale_list)

            # Get the latest and newest times.
            if (max_time < tmp_data[gl.COLUMN_TIMESTAMP].max()):
                max_time = tmp_data[gl.COLUMN_TIMESTAMP].max()
            if (min_time > tmp_data[gl.COLUMN_TIMESTAMP].min()):
                min_time = tmp_data[gl.COLUMN_TIMESTAMP].min()

            # Append the data to the output arrays.
            if all_data is None:
                all_data = tmp_data.copy()
                all_data_d = np.array(tmp_data.shape[0] * [day])

                ellipse_data = tmp_data[cut_array]
                ellipse_data_d = np.array(ellipse_data.shape[0] * [day])

                non_ellipse_data = all_data[~cut_array]
                non_ellipse_data_d = np.array(non_ellipse_data.shape[0] * [day])
            else:
                all_data = all_data.append(tmp_data)
                all_data_d = np.append(all_data_d,
                    (all_data.shape[0] - all_data_d.shape[0]) * [day])

                ellipse_data = ellipse_data.append(tmp_data[cut_array])
                ellipse_data_d = np.append(ellipse_data_d,
                    (ellipse_data.shape[0] - ellipse_data_d.shape[0]) * [day])

                non_ellipse_data = non_ellipse_data.append(tmp_data[~cut_array])
                non_ellipse_data_d = np.append(
                    non_ellipse_data_d, (non_ellipse_data.shape[0] -
                        non_ellipse_data_d.shape[0]) * [day])

    # Calculate the measurement duration.
    duration = (max_time - min_time) * 4. / 1e9
    # Calculate the number of pixel-days.
    num_of_days = duration * num_of_pixels / (3600 * 24)

    # TODO: Instead of returning that many arrays, return a class.

    return all_data, all_data_d, ellipse_data, ellipse_data_d, \
            non_ellipse_data, non_ellipse_data_d, num_of_days, duration