import holoviews as hv
import bokeh
hv.extension('bokeh', logo=False)
import numpy as np
import os
import sys
import importlib
import matplotlib
sys.path.append(os.path.abspath("."))
import Formatter
importlib.reload(Formatter)
from Formatter import *
from warnings import warn
from IPython.display import display_html
from holoviews.operation.datashader import datashade, dynspread, spread

# TODO: Combine set_filed functions.
# TODO: Generalize get_parameters.


class Overlay:
    """ A customization of the holoviews.core.Overlay class.

        Methods:
            copy(): Create a copy of this object.
            get_parameters(): Get the object's attributes.
            options(name=None, **kwargs): Modify the attributes.
            opts(name=None, **kwargs): Modify the drawing options.
            plot(): Show the figure.
            save(path): Save the figure.
    """

    # TODO: Combine the save and plot methods.

    def __init__(
        self, log=False, norm=1., xlabel="x", ylabel="y",
            xlim=None, ylim=None, grid=True, legendpos=False):
        """
        Args:
            log (bool, optional): If the y-axis should be scaled logarithmical.
                                  Defaults to False.
            norm (float, optional): The normalization of the y-values. Defaults to 1.
            xlabel (str, optional): The label of the x-axis. Defaults to "x".
            ylabel (str, optional): The label of the y-axis. Defaults to "y".
            xlim (tuple, optional): The x range. Defaults to None.
            ylim (tuple, optional): The y range. Defaults to None.
            grid (bool, optional): If the grid should be shown. Defaults to True.
            legendpos (bool, optional): If the legend should be positioned to
                                        the right side of the figure. If False,
                                        it is placed in the figure. Defaults to False.
        """

        # Define the used color list.
        self._color_list = bokeh.palettes.Category10_10

        # Define the label text and tick size of the axes.
        self._label_size = 20.
        self._tick_size = 18.
        # Define the figure dimensions.
        self._plot_height = 636
        self._plot_width = 900

        self._fontsize={'legend': 18, 'labels': self._label_size,
                       'ticks': self._tick_size}

        self._grid_style = {'grid_line_color': 'black', 'grid_line_width': 1.5,
                            'minor_xgrid_line_color': 'lightgray',
                            'minor_ygrid_line_color': 'lightgray',
                            'grid_line_alpha': 0.3, 'minor_ygrid_line_alpha': 0.5}

        # Assign the input arguments.
        self._log = log
        self._norm = norm
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._xlim = xlim
        self._ylim = ylim
        self._grid = grid
        self._legendpos = legendpos

        # Initialize the layouts
        self._layouts = {}


    def get_parameters(self):
        """ Get the attributes of this object.

        Returns:
            dict: A dictionary of attributes.
        """

        return {"log": self._log, "norm": self._norm, "xlabel": self._xlabel,
                "ylabel": self._ylabel, "xlim": self._xlim, "ylim": self._ylim,
                "grid": self._grid, "legendpos": self._legendpos}


    def _set_limits(self):
        """ Apply the x- and y-ranges to all layouts. """

        # Iterate over all layouts and set the x- and y-ranges.
        for name, layout in self._layouts.items():
            if (self._xlim is None) and (layout._min_x is not None) and \
                    (layout._max_x is not None):
                # The layout has a custom x-range, which will be applied.
                layout._layout = \
                    layout._layout.redim.range(
                        x=(layout._min_x, layout._max_x))
            elif (self._xlim is not None):
                # Apply the global x-range.
                layout._layout = \
                    layout._layout.redim.range(x=self._xlim)

            if (self._ylim is None) and (layout._min_y is not None) and \
                    (layout._max_y is not None):
                # The layout has a custom y-range, which will be applied.
                layout._layout = \
                    layout._layout.redim.range(
                        y=(layout._min_y, layout._max_y * 1.1))
            elif (self._ylim is not None):
                # Apply the global y-range.
                layout._layout = \
                    layout._layout.redim.range(y=self._ylim)


    def options(self, name=None, **kwargs):
        """ Change the attributes of this object and connected layouts.

        Args:
            name (string, optional): The name of the layout, which should be
                                     modified. None, if all layouts should be
                                     affected. Defaults to None.

        Returns:
            Overlay: Returns a modified version of itself.
        """

        # The names of the attributes.
        name_list = ["log", "norm", "xlim", "ylim", "xlabel", "ylabel", "grid",
                     "legendpos"]

        def set_field(prop, key):
            """ Get the attribute value stored in kwargs.

            Args:
                prop (attribute value): The current attribute value.
                key (attribute name): The name of the attribute.

            Returns:
                attribute value: The new attribute value.
            """

            # If the name of the attribute is in kwargs, get the new value and
            # remove it from kwargs.
            if key in kwargs:
                value = kwargs[key]
                kwargs.pop(key, None)
                return value
            else:
                # The attribute should not be changed, return its value.
                return prop

        # Modify each attribute.
        self._log = set_field(self._log, name_list[0])
        self._norm = set_field(self._norm, name_list[1])
        self._xlim = set_field(self._xlim, name_list[2])
        self._ylim = set_field(self._ylim, name_list[3])
        self._xlabel = set_field(self._xlabel, name_list[4])
        self._ylabel = set_field(self._ylabel, name_list[5])
        self._legendpos = set_field(self._legendpos, name_list[6])

        # Adjust the log attribute of all layouts.
        for l_name, layout in self._layouts.items():
            layout._options["logy"] = self._log

        for key, value in kwargs.items():
            for l_name, layout in self._layouts.items():
                if name is None:
                    # Change the attributes of all layouts.
                    layout._options[key] = value
                elif name == l_name:
                    # Change the attributes of the given layouts.
                    layout._options[key] = value

        return self


    def opts(self, name=None, **kwargs):
        """ Modifies the draw options of the layouts.

        Args:
            name (string, optional): The name of the layout, which should be
                                     modified. None, if all layouts should be
                                     affected. Defaults to None.

        Returns:
            Overlay: Returns a modified version of itself.
        """

        for key, value in kwargs.items():
            for l_name, layout in self._layouts.items():
                if name is None:
                    # Change the draw options of all layouts.
                    layout._opts[key] = value
                elif name == l_name:
                    # Change the draw options of the given layout.
                    layout._opts[key] = value

        return self


    def plot(self):
        """ Show the figure. """

        # Apply the x- and y-ranges.
        self._set_limits()

        # Initialize the figure.
        out = None

        # Iterate over all layouts and apply the drawing options.
        # Add the layouts to the output figure.
        for name, layouts in self._layouts.items():
            if type(layouts._layout) == hv.core.overlay.Overlay:
                for layout in layouts._layout:
                    if out is None:
                        out = layout.opts(**layouts._opts).options(**layouts._options)
                    else:
                        out *= layout.opts(**layouts._opts).options(**layouts._options)
            else:
                if out is None:
                    out = layouts._layout.opts(**layouts._opts).options(**layouts._options)
                else:
                    out *= layouts._layout.opts(**layouts._opts).options(**layouts._options)

        # Set the axes formatter and the figure dimensions.
        out = self._apply_layout(out)

        # Apply the formatter for the x-axis and the grid if defined.
        if self._grid:
            out = out.opts(
                xformatter=formatter, gridstyle=self._grid_style, show_grid=True)
        else:
            out = out.opts(xformatter=formatter)

        # Apply the formatter for the y-axis.
        if self._log:
            out = out.opts(yformatter=log_formatter)
        else:
            out = out.opts(yformatter=formatter)

        # Set the position of the legend.
        if self._legendpos:
            out = out.opts(legend_position='right', legend_limit=100)

        # Show the plot.
        hv.output(out)

        # renderer = hv.renderer('bokeh')
        # widget = renderer.get_widget(out, 'widgets')
        # html = renderer.static_html(out)
        # display_html(html)

    def _add_layout(self, name, layout):
        """ Add a layout to this object.

        Args:
            name (string): The name of the layout.
            layout (Layout): The layout
        """

        self._layouts[name] = layout
        layout._options["logy"] = self._log


    def _apply_layout(self, layout):
        """ Set the drawing style of the layout.
        Args:
            layout (Layout): A layout.
        Returns:
            Layout: A modified copy of the layout.
        """

        return layout.opts(
            xformatter=formatter, fontsize=self._fontsize,
            width=self._plot_width, height=self._plot_height).redim.label(
            x=self._xlabel, y=self._ylabel)


    def __mul__(self, other):
        """ Combine two Overlay objects.

        Args:
            other (Overlay): Another Overlay object.

        Returns:
            Overlay: The merged Overlay objects.
        """

        # Create a copy of itself.
        out = self.copy()
        # Generate copies of the layouts and assign them to the copied overlay.
        for name, layout in self._layouts.items():
            temp = layout.copy(overlay=out)

        # Add copies of the layouts of the other overlay to the copied overlay.
        if type(other) is type(self):
            for name, layout in other._layouts.items():
                temp = layout.copy(overlay=out)
        else:
            for name, layout in other._overlay._layouts.items():
                temp = layout.copy(overlay=out)

        return out


    def copy(self):
        """
        Returns:
            Overlay: A copy from itself.
        """
        return Overlay(**self.get_parameters())


    def save(self, path):
        """ Save the figure at the given path.

        Args:
            path (string): Path to the file, which should consists of the figure.
        """

        # Apply the x- and y-ranges.
        self._set_limits()

        # Initialize the figure.
        out = None

        # Iterate over all layouts and apply the drawing options.
        # Add the layouts to the output figure.
        for name, layouts in self._layouts.items():
            if type(layouts._layout) == hv.core.overlay.Overlay:
                for layout in layouts._layout:
                    if out is None:
                        out = layout.opts(**layouts._opts).options(**layouts._options)
                    else:
                        out *= layout.opts(**layouts._opts).options(**layouts._options)
            else:
                if out is None:
                    out = layouts._layout.opts(**layouts._opts).options(**layouts._options)
                else:
                    out *= layouts._layout.opts(**layouts._opts).options(**layouts._options)

        # Set the axes formatter and the figure dimensions.
        out = self._apply_layout(out)

        # Apply the formatter for the x-axis and the grid if defined.
        if self._grid:
            out = out.opts(
                xformatter=formatter, gridstyle=self._grid_style, show_grid=True)
        else:
            out = out.opts(xformatter=formatter)

        # Apply the formatter for the y-axis.
        if self._log:
            out = out.opts(yformatter=log_formatter)
        else:
            out = out.opts(yformatter=formatter)

        # Save the figure.
        hv.save(out, path, resources='inline', backend="bokeh")


class Layout:
    """ A customization of the holoviews.core.Layout class.

        Methods:
            copy(): Create a copy of this object.
            get_parameters(): Get the object's attributes.
            options(**kwargs): Modify the attributes.
            opts(**kwargs): Modify the drawing options.
            plot(): Show the figure.
            save(path): Save the figure.
    """

    def __init__(
        self, alpha=1., color=0, title="", layout=None,
            name=None, overlay=None, opts=None, options=None, **kwargs):
        """
        Args:
            alpha (float, optional): The transparency. A value between 0 and 1.
                                     Defaults to 1.
            color (int, optional): The color index. A value between 0 and 9.
                                   Defaults to 0.
            title (str, optional): The title of the figure. Defaults to "".
            layout (holoviews.Layout, optional): The corresponding
                                                 holoviews.Layout. Defaults to None.
            name (string, optional): The name of the object. Defaults to None.
            overlay (Overlay, optional): The parent overlay object. Defaults to None.
            opts (dict, optional): The drawing options. Defaults to None.
            options (dict, optional): The display options. Defaults to None.
        """

        # Set or initialize the drawing optios.
        if opts is None:
            self._opts = {}
        else:
            self._opts = opts

        if options is None:
            self._options = {}
        else:
            self._options = options

        # Add this object to an existing overlay object or create a new one.
        if overlay is None:
            # Set the name if not set.
            self._name = "Layout_0"
            if not (name is None):
                self._name = name
            # Create an overlay object.
            self._overlay = Overlay(**kwargs)
            # Add this object to the overlay.
            self._overlay._add_layout(self._name, self)
        else:
            self._overlay = overlay
            # Set the name if not set.
            self._name = "Layout_" + str(len(self._overlay._layouts))
            if not (name is None):
                if not (name in self._overlay._layouts):
                    self._name = name
            # Add this object to the overlay.
            self._overlay._add_layout(self._name, self)

        # Define the transparency.
        self._alpha = alpha
        # Define the color.
        self._color = self._set_color(color)
        # Set the title.
        self._title = title
        # The holoviews.Layout.
        self._layout = layout

        # Define ranges for the x- and y-axes.
        self._min_x = None
        self._min_y = None
        self._max_x = None
        self._max_y = None


    def get_parameters(self):
        """ Get the attributes of this object.

        Returns:
            dict: A dictionary of attributes.
        """

        return {"alpha": self._alpha, "color": self._color,
                "title": self._title, "name": self._name}


    def options(self, **kwargs):
        """ Change the attributes of this object.

        Returns:
            Layout: Returns a modified version of itself.
        """

        # The names of the attributes.
        name_list = ["alpha", "color", "title", "label"]

        def set_field(prop, key):
            """ Get the attribute value stored in kwargs.

            Args:
                prop (attribute value): The current attribute value.
                key (attribute name): The name of the attribute.

            Returns:
                attribute value: The new attribute value.
            """

            # If the name of the attribute is in kwargs, get the new value and
            # remove it from kwargs.
            if key in kwargs:
                if key == name_list[1]:
                    value = self._set_color(color=kwargs[key])
                else:
                    value = kwargs[key]
                kwargs.pop(key, None)
                return value
            else:
                # The attribute should not be changed, return its value.
                return prop

        # Modify each attribute.
        self._alpha = set_field(self._alpha, name_list[0])
        self._color = set_field(self._color, name_list[1])
        self._title = set_field(self._title, name_list[2])
        self._title = set_field(self._title, name_list[2])


        _ = self._overlay.options(self._name, **kwargs)

        return self


    def opts(self, **kwargs):
        """ Change drawing options.

        Returns:
            Layout: A modified version of itself.
        """

        _ = self._overlay.opts(self._name, **kwargs)
        return self


    def _set_limits(self, x, y):
        """ Apply the ranges of the x- and y-axes.

        Args:
            x (tuple): Boundaries of the x range.
            y (tuple): Boundaries of the y range.
        """

        # Apply the boundaries.
        if self._min_x is None:
            self._min_x = x.min()
            self._max_x = x.max()
            self._min_y = y.min()
            self._max_y = y.max()
        # Check if the set boundaries are smaller than the new ones.
        # In this case use the new boundaries.
        else:
            if self._min_x > x.min():
                self._min_x = x.min()
            if self._min_y > y.min():
                self._min_y = y.min()
            if self._max_x < x.max():
                self._max_x = x.max()
            if self._max_y < y.max():
                self._max_y = y.max()


    def _reset_limits(self):
        """ Reset the boundaries of the x- and y-axes. """

        self._layout = None
        self._min_x = None
        self._min_y = None
        self._max_x = None
        self._max_y = None


    def _set_color(self, color):
        """ Get the color corresponding to the index.

        Args:
            color (int or color-like): Either an index defining a color or a color.

        Returns:
            bokeh.colors.Color: A color.
        """

        # Check if a color or an integer is given.
        if type(color) != int:
            # Return the color.
            return color

        # Return a color of the color map.
        i = color
        while i > 9:
            # Only 10 colors are inside the color map.
            i -= 10
        return self._overlay._color_list[i]


    def _mod_y(self, y):
        """ Scale the y-data with the norm and remove zeros.

        Args:
            y (numpy.array): Modified y-data.
        """

        def set_zeros(zeros, y):
            """ Change zeros to lower limit values.

            Args:
                zeros (numpy.array): The position of the zeros and y.
                y (numpy.array): The data.

            Returns:
                numpy.array: Switched zero values.
            """

            # The lower limit.
            lim = 1e-2
            # Use this modifier instead of the lower limit, if data contains
            # lower values than the lower limit.
            mod = 10.
            if len(y) == 0:
                # Nothing to show.
                return y

            if len(y[y > 0]) == 0:
                # Only zeros values are in the data.
                return y

            if lim > y[y > 0].min() / mod:
                # There are smaller values in y than the lower limit. Set the
                # lower limit to 10 % of the smallest value.
                out = zeros * y[y > 0].min() / mod
            else:
                out = zeros * lim
            return out

        # Normalize the data.
        out = y / self._overlay._norm
        # Get the zeros in data.
        zeros = (out == 0).astype(np.float)
        # Change the zeros.
        return out + set_zeros(zeros, out)


    def plot(self):
        """ Show the figure. """

        self._overlay.plot()

    def save(self, path):
        """ Save the figure.

        Args:
            path (string): Name of the file, which should contain the figure.
        """

        self._overlay.save(path)


    def copy(self):
        """ Create a copy of itself.

        Returns:
            Layout: A copy of itself.
        """

        return Layout(
            alpha=self._alpha, color=self._color, title=self._title,
            layout=self._layout, name=self._name, opts=self._opts, options=self._options)


    def __mul__(self, other):
        """ Combine two layouts.

        Args:
            other (Layout or Overlay): A second layout object.

        Returns:
            Overlay: A combined overlay object.
        """

        if type(other) != type(Overlay()):
            return self._overlay.__mul__(other._overlay)
        else:
            return self._overlay.__mul__(other)


class Histogram(Layout):
    """ A customization of the holoviews.Histogram class.

        Methods:
            copy(): Create a copy of this object.
            get_parameters(): Get the object's attributes.
    """

    def __init__(self, data, ydata=None, range=None, bins=10, **kwargs):
        """
        Args:
            data (dict or numpy.array): Either the (a dictionary of) raw data,
                                        which should be histogrammed, or the
                                        x-axis of the histogrammed data.
            ydata (numpy.array, optional): The histogrammed y-axis. Defaults to None.
            range (tuple, optional): The boundaries of the histogram. Will not
                                     be used if ydata is set. Defaults to None.
            bins (int, optional): The number of bins. Will not be used if ydata
                                  is set. Defaults to 10.
        """

        # Assign hte attributes.
        super(Histogram, self).__init__(**kwargs)
        self._data = data
        self._ydata = ydata
        self._range = range
        self._bins = bins

        # Define the graph colors.
        self._options["line_color"] = self._color
        self._options["fill_color"] = self._color
        self._options["line_alpha"] = self._alpha
        self._options["fill_alpha"] = self._alpha

        # Set the title.
        if (self._title is not None) and ("\n" in self._title):
            left_title = self._title.split("\n")[0]
            right_title = self._title.split("\n")[1]
            temp_alpha = self._alpha
            self._title = right_title
            self._alpha = 0.
            copy_hist = self.copy(overlay=self._overlay)
            self._title = left_title
            self._alpha = temp_alpha


        self._set_layout()

    def get_parameters(self):
        """ Get the attributes of this object.

        Returns:
            dict: A dictionary of attributes.
        """

        out = {}
        out["ydata"] = self._ydata
        out["range"] = self._range
        out["bins"] = self._bins
        for key, value in super(Histogram, self).get_parameters().items():
            out[key] = value
        return out


    def _get_axes(self, data, ydata):
        """ Calculate the histogram data and remove all zero values.

        Args:
            data (dict or numpy.array): Either the (a dictionary of) raw data,
                                        which should be histogrammed, or the
                                        x-axis of the histogrammed data.
            ydata (numpy.array, optional): The histogrammed y-axis.
        """

        def mod_log(x, y):
            """ Find the minimum value of y and add this value to the end and
                start of the y-axis.

            Args:
                x (numpy.array): x-values
                y (numpy.array): y-values

            Returns:
                numpy.array, numpy.array: The modified x- and y-axes.
            """

            # Increase the size of the x-axis. This is done because the
            # holoviews.Area class is used instead of holoviews.Histogram.
            # The first and last entries of y have to be filled with the smallest
            # values, that is why the lengths of the y-axis and as well of the
            # x-axis are increased.
            out_x =  np.append(np.array([x[0]]), x)
            if x.shape[0] == y.shape[0]:
                # If the edge axis of numpy.histogram is used as input for x,
                # the length of the value axis and edge axis differ by one.
                out_x = np.append(out_x, x[-1])

            # Set a lower value limit.
            if y.min() > 1e-2:
                min_val = 1e-2
            else:
                min_val = y.min()

            # Increase the length of y.
            out_y = np.append(y, min_val)
            out_y = np.append(np.array([min_val]), out_y)
            return out_x, out_y

        if ydata is None:
            # Create the histogram.
            y, x = np.histogram(data, bins=self._bins, range=self._range)
        else:
            x = data.copy()
            y = ydata.copy()

        # Remove zero values. Necessary for logarithmic scales.
        y = self._mod_y(y)

        # Increase the lengths of both axes.
        x, y = mod_log(x, y)

        return x, y


    def _get_histogram(self, x, y, label, color=None, alpha=None):
        """ Generate the histogram plot.

        Args:
            x (numpy.array): The x-values.
            y (numpy.array): The y-values.
            label (string): The label of this data set.
            color (bokeh.Color, optional): The color of the histogram. Defaults to None.
            alpha (float, optional): The transparency of the plot. A value
                                     between 0 and 1. Defaults to None.
        """

        # Plot the data points and connect them by steps.
        if label is None:
            out = hv.Scatter((x.min(), y.min())).options(
                **self._options).opts(marker="s", **self._opts)
        else:
            out = hv.Scatter((x.min(), y.min()), label=label).options(
                **self._options).opts(marker="s", **self._opts)

        # Set the color.
        if color is not None:
            out = out.options(line_color=color, fill_color=color)

        # Create data points for the filling.
        x_area = np.zeros(x.shape[0] * 2)
        y_area = np.zeros(y.shape[0] * 2)

        for i in range(x.shape[0]):
            x_area[2 * i] = x[i]
            x_area[2 * i + 1] = x[i]

            if i > 0:
                y_area[2 * i] = y[i - 1]
            else:
                y_area[2 * i] = y[i]

            y_area[2 * i + 1] = y[i]

        # Plot the filling.
        area_tmp = hv.Area((x_area, y_area)).options(**self._options)

        # Adjust the color and transparency of the filling.
        if color is not None:
            area_tmp = area_tmp.options(fill_color=color, line_color=color)
        if alpha is not None:
            area_tmp = area_tmp.options(fill_alpha=alpha)

        # The line is already defined before.
        area_tmp = area_tmp.options(line_alpha=0.)

        # Add the filling.
        out *= area_tmp

        # Assign the plot to the layout.
        if self._layout is None:
            self._layout = out
        else:
            self._layout *= out

        # Adjust the axes boundaries.
        self._set_limits(x, y)


    def _set_layout(self):
        """ Create the figure. """

        # Initialize the boundaries of the axes.
        self._reset_limits()


        if ((type(self._data) != dict) and (type(self._ydata) != dict)):
            # Generate the single histogram.
            x, y = self._get_axes(self._data, self._ydata)
            self._get_histogram(x, y, self._title)

        elif self._ydata is None:
            # The x-axes in data share the same y-axis.
            # TODO: Is there any applications for this? Should be removed.
            # Maybe change data and ydata in this case.

            # Iterate over all x-axes.
            # Define the transparency and color by the index.
            counter = 0
            # TODO: Use enumerate.
            for d in self._data:
                x, y = self._get_axes(self._data[d], self._ydata)
                self._get_histogram(x, y, d, color=self._set_color(counter),
                                    alpha=self._alpha * (1. - counter / (len(self._data) + 1)))
                counter += 1

        elif (type(self._data) != dict):
            # The x-axis is the same for all data sets in ydata.

            # Iterate over all data sets.
            # Define the transparency and color by the index.
            counter = 0
            # TODO: Use enumerate.
            for d in self._ydata:
                x, y = self._get_axes(self._data, self._ydata[d])
                self._get_histogram(x, y, d, color=self._set_color(counter),
                                    alpha=self._alpha * (1. - counter / (len(self._ydata) + 1)))
                counter += 1

        else:
            # Each data set has its own x-axis.

            # Iterate over all data sets.
            # Define the transparency and color by the index.
            counter = 0
            if len(self._data) != len(self._ydata):
                print("Not matching shapes")
            else:
                # TODO: Use enumerate.
                for d in self._data:
                    x, y = self._get_axes(self._data[d], self._ydata[d])
                    self._get_histogram(x, y, d, color=self._set_color(counter),
                                        alpha=self._alpha * (1. - counter / (len(self._data) + 1)))
                    counter += 1

        # Assign the figure.
        self._layout = self._overlay._apply_layout(self._layout)


    def copy(self, overlay=None):
        """ Create a copy of itself.

        Args:
            overlay (Overlay, optional): The parent overlay class to which the
                                         copy should be assigned. None, if a new
                                         one should be created. Defaults to None.

        Returns:
            Histogram: A copy of itself.
        """

        return Histogram(self._data, **self.get_parameters(), overlay=overlay,
                         opts=self._opts, options=self._options)


class Curve(Layout):
    """ A customization of the holoviews.Curve class.

        Methods:
            copy(): Create a copy of this object.
            get_parameters(): Get the object's attributes.
            options(**kwargs): Modify the attributes.
    """

    def __init__(self, xaxis, data, line_width=2., interpolation='linear', **kwargs):
        """
        Args:
            xaxis (dict or numpy.array): (A dictionary of) x-values.
            data (dict or numpy.array): (A dictionary of) y-values.
            line_width (float, optional): The line width. Defaults to 2.
            interpolation (str, optional): The interpolation method.
                                           Defaults to 'linear'.
        """

        # Assign the attributes.
        super(Curve, self).__init__(**kwargs)
        self._xaxis = xaxis
        self._data = data
        self._line_width = line_width
        self._options["line_color"] = self._color
        self._options["line_width"] = self._line_width
        self._options["alpha"] = self._alpha
        self._interpolation = interpolation
        self._options["interpolation"] = self._interpolation

        # Generate the holoviews.Layout object.
        self._set_layout()


    def get_parameters(self):
        """ Get the attributes of this object.

        Returns:
            dict: A dictionary of attributes.
        """

        out = {}
        out["line_width"] = self._line_width
        out["interpolation"] = self._interpolation
        for key, value in super(Curve, self).get_parameters().items():
            out[key] = value
        return out


    def _get_curve(self, x, y, label, color=None):
        """ Generate the curve plot.

        Args:
            x (numpy.array): The x-values.
            y (numpy.array): The y-values.
            label (string): The label of this data set.
            color (bokeh.Color, optional): The color of the histogram. Defaults to None.
        """

        # Plot the curve.
        out = hv.Curve((x, y), label=label).options(
            **self._options).opts(**self._opts)
        # Assign the color.
        if color is not None:
            out = out.options(line_color=color)

        # Assign the plot to the layout.
        if self._layout is None:
            self._layout = out
        else:
            self._layout *= out

        # Adjust the axes boundaries.
        self._set_limits(x, y)


    def _set_layout(self):
        """ Create the figure. """

        # Initialize the boundaries of the axes.
        self._reset_limits()

        if type(self._data) != dict:
            # Generate a single curve.
            yaxis = self._mod_y(self._data)
            self._get_curve(self._xaxis, yaxis, self._title)

        elif type(self._xaxis) != dict:
            # The x-axis is the same for all data sets in data.

            # Iterate over all data sets.
            # Define the transparency and color by the index.
            counter = 0
            # TODO: Use enumerate.
            for d in self._data:
                yaxis = self._mod_y(self._data[d])
                self._get_curve(
                    self._xaxis, yaxis, d, color=self._set_color(counter))
                counter += 1
        else:
            # Each data set has its own x-axis.

            # Iterate over all data sets.
            # Define the transparency and color by the index.
            counter = 0
            # TODO: Use enumerate.
            for d in self._data:
                yaxis = self._mod_y(self._data[d])
                self._get_curve(
                    self._xaxis[d], yaxis, d, color=self._set_color(counter))
                counter += 1

        # Assign the figure.
        self._layout = self._overlay._apply_layout(self._layout)


    def options(self, **kwargs):
        """ Change the attributes of this object.

        Returns:
            Curve: Returns a modified version of itself.
        """

        # The names of the attributes.
        name_list = ["line_width", "interpolation"]

        def set_field(prop, key):
            """ Get the attribute value stored in kwargs.

            Args:
                prop (attribute value): The current attribute value.
                key (attribute name): The name of the attribute.

            Returns:
                attribute value: The new attribute value.
            """

            if key in kwargs:
                value = kwargs[key]
                kwargs.pop(key, None)
                return value
            else:
                return prop

        self._line_width = set_field(self._line_width, name_list[0])
        return super(Curve, self).options(**kwargs)


    def copy(self, overlay=None):
        """ Create a copy of itself.

        Args:
            overlay (Overlay, optional): The parent overlay class to which the
                                         copy should be assigned. None, if a new
                                         one should be created. Defaults to None.

        Returns:
            Curve: A copy of itself.
        """

        return Curve(
            self._xaxis, self._data, **self.get_parameters(),
            overlay=overlay, opts=self._opts, options=self._options)


class VLine(Layout):
    """ A customization of the holoviews.VLine class.

        Methods:
            copy(): Create a copy of this object.
            get_parameters(): Get the object's attributes.
            options(**kwargs): Modify the attributes.
    """

    def __init__(self, positions, line_width=2., **kwargs):
        """
        Args:
            positions (number, list or dict): The x-positions, where the lines
                                              should be drawn,
            line_width (float, optional): The line width. Defaults to 2.
        """

        # Assign the attributes.
        super(VLine, self).__init__(**kwargs)
        self._positions = positions
        self._line_width = line_width
        self._options["line_color"] = self._color
        self._options["line_width"] = self._line_width
        self._options["alpha"] = self._alpha

        # Generate the holoviews.Layout object.
        self._set_layout()


    def get_parameters(self):
        """ Get the attributes of this object.

        Returns:
            dict: A dictionary of attributes.
        """

        out = {}
        out["line_width"] = self._line_width
        for key, value in super(VLine, self).get_parameters().items():
            out[key] = value
        return out


    def _get_vlines(self, x, color, label):
        """ Generate the lines.

        Args:
            x (number): Position of the line on the x-axis.
            color (bokeh.Color): The color of the line.
            label (string): The name of the line.
        """

        # Plot the line.
        out = hv.VLine(x, label=label).options(
            **self._options).opts(**self._opts)

        # Assign the plot to the layout.
        if self._layout is None:
            self._layout = out
        else:
            self._layout *= out


    def _set_layout(self):
        """ Create the figure. """

        # Initialize the boundaries of the axes.
        self._reset_limits()

        if (type(self._positions) != dict) and \
           (type(self._positions) != list) and \
                (type(self._positions) != type(np.array(0))):
            # Generate a single line.
            self._get_vlines(
                self._positions, self._color, self._title)
        else:
            if (type(self._positions) != dict):
                # Generate multiple lines with the same label.
                for pos in self._positions:
                    self._get_vlines(
                        pos, self._color, self._title)
            else:
                # Generate multiple lines with own labels.
                for pos in self._positions:
                    self._get_vlines(
                        self._positions[pos], self._color, pos)

        # Assign the figure.
        self._layout = self._overlay._apply_layout(self._layout)


    def options(self, **kwargs):
        """ Change the attributes of this object.

        Returns:
            VLine: Returns a modified version of itself.
        """

        name_list = ["line_width"]

        def set_field(prop, key):
            """ Get the attribute value stored in kwargs.

            Args:
                prop (attribute value): The current attribute value.
                key (attribute name): The name of the attribute.

            Returns:
                attribute value: The new attribute value.
            """

            if key in kwargs:
                value = kwargs[key]
                kwargs.pop(key, None)
                return value
            else:
                return prop

        self._options["line_width"] = set_field(
            self._options["line_width"], name_list[0])
        return super(VLine, self).options(**kwargs)


    def copy(self, overlay=None):
        """ Create a copy of itself.

        Args:
            overlay (Overlay, optional): The parent overlay class to which the
                                         copy should be assigned. None, if a new
                                         one should be created. Defaults to None.

        Returns:
            VLine: A copy of itself.
        """

        return VLine(
            self._positions, **self.get_parameters(),
            overlay=overlay, opts=self._opts, options=self._options)


class Text(Layout):
    """ A customization of the holoviews.Text class.

        Methods:
            copy(): Create a copy of this object.
            get_parameters(): Get the object's attributes.
            options(**kwargs): Modify the attributes.
    """

    def __init__(self, xpos, ypos, label, angle=0., **kwargs):
        """
        Args:
            xpos (number): The x coordinate of the text field.
            ypos (number): The y coordinate of the text field.
            label (string): The text.
            angle (number, optional): The rotation of the text field. Defaults to 0.
        """

        # Assign the attributes.
        super(Text, self).__init__(**kwargs)
        self._xpos = xpos
        self._ypos = ypos
        self._title = label
        self._options["angle"] = angle
        self._options["text_color"] = self._color
        self._options["text_alpha"] = self._alpha
        self._options["text_font_size"] = "20px"

        # Generate the holoviews.Layout object.
        self._set_layout()


    def get_parameters(self):
        """ Get the attributes of this object.

        Returns:
            dict: A dictionary of attributes.
        """

        out = {}
        for key, value in super(Text, self).get_parameters().items():
            out[key] = value
        return out


    def _get_text(self, x, y, color, label):
        """ Generate the text field.

        Args:
            x (number): The x coordinate of the text field.
            y (number): The y coordinate of the text field.
            color (bokeh.Color): The color of the text.
            label (string): The text.
        """

        out = hv.Text(x, y, label).options(
            **self._options).opts(**self._opts)
        self._layout = out


    def _set_layout(self):
        """ Create the figure. """

        self._get_text(self._xpos, self._ypos, self._color, self._title)
        self._layout = self._overlay._apply_layout(self._layout)


    def options(self, **kwargs):
        """ Change the attributes of this object.

        Returns:
            Text: Returns a modified version of itself.
        """

        name_list = ["angle"]

        def set_field(prop, key):
            """ Get the attribute value stored in kwargs.

            Args:
                prop (attribute value): The current attribute value.
                key (attribute name): The name of the attribute.

            Returns:
                attribute value: The new attribute value.
            """

            if key in kwargs:
                value = kwargs[key]
                kwargs.pop(key, None)
                return value
            else:
                return prop

        self._options["angle"] = set_field(
            self._options["angle"], name_list[0])
        return super(Text, self).options(**kwargs)


    def copy(self, overlay=None):
        """ Create a copy of itself.

        Args:
            overlay (Overlay, optional): The parent overlay class to which the
                                         copy should be assigned. None, if a new
                                         one should be created. Defaults to None.

        Returns:
            Text: A copy of itself.
        """

        return Text(
            self._xpos, self._ypos, self._title, **self.get_parameters(),
            overlay=overlay, opts=self._opts, options=self._options)


class Datashade(Layout):
    """ A customization of the holoviews.operation.datashader.datashade class.

        Methods:
            copy(): Create a copy of this object.
            get_parameters(): Get the object's attributes.
            options(**kwargs): Modify the attributes.
    """

    def __init__(self, xpos, ypos, dynamic=True, **kwargs):
        """
        Args:
            xpos (numpy.array): The x-values.
            ypos (numpy.array): The y-values.
            dynamic (bool, optional): Defines if the density should be
                                      calculated dynamically. Meaning, if it
                                      should be recalculated by zooming in/out.
                                      Defaults to True.
        """

        # Assign the attributes.
        super(Datashade, self).__init__(**kwargs)
        self._xpos = xpos
        self._ypos = ypos
        self._dynamic = dynamic

        # Generate the holoviews.Layout object.
        self._set_layout()


    def get_parameters(self):
        """ Get the attributes of this object.

        Returns:
            dict: A dictionary of attributes.
        """

        out = {}
        out["dynamic"] = self._dynamic
        for key, value in super(Datashade, self).get_parameters().items():
            out[key] = value
        return out


    def _get_datashade(self, x, y, label):
        """ Generate the density plot.

        Args:
            x (numpy.array): The x-values.
            y (numpy.array): The y-values.
            label (string): The label of the plot.
        """

        # Generate a custom color map.
        # TODO: Has to be changed according the matplotlib warnings.
        my_palette = [(0, 0, 0)]
        for i in bokeh.palettes.Spectral11:
            my_palette.append(matplotlib.colors.hex2color(i))
        my_palette = my_palette[1:]
        my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "", my_palette)

        # Check if the density should be calculated dynamically.
        # It must not be dynamically, if one want to save the figure automatically.
        if self._dynamic:
            # Generate the density plot.
            out = spread(datashade(
                hv.Points((x, y), label=label), cmap=my_cmap,
                dynamic=self._dynamic), px=2).options(
                    **self._options).opts(**self._opts)
        else:
            out = datashade(
                hv.Points((x, y), label=label), cmap=my_cmap,
                dynamic=self._dynamic).options(
                    **self._options).opts(**self._opts)


        self._layout = out


    def _set_layout(self):
        """ Create the figure. """

        self._get_datashade(self._xpos, self._ypos, self._title)
        self._layout = self._overlay._apply_layout(self._layout)


    def options(self, **kwargs):
        """ Change the attributes of this object.

        Returns:
            Datashade: Returns a modified version of itself.
        """

        name_list = ["dynamic"]

        def set_field(prop, key):
            """ Get the attribute value stored in kwargs.

            Args:
                prop (attribute value): The current attribute value.
                key (attribute name): The name of the attribute.

            Returns:
                attribute value: The new attribute value.
            """

            if key in kwargs:
                value = kwargs[key]
                kwargs.pop(key, None)
                return value
            else:
                return prop

            self._options["dynamic"] = set_field(
                self._options["dynamic"], name_list[0])

        return super(Datashade, self).options(**kwargs)


    def copy(self, overlay=None):
        """ Create a copy of itself.

        Args:
            overlay (Overlay, optional): The parent overlay class to which the
                                         copy should be assigned. None, if a new
                                         one should be created. Defaults to None.

        Returns:
            Datashade: A copy of itself.
        """

        return Datashade(
            self._xpos, self._ypos, **self.get_parameters(),
            overlay=overlay, opts=self._opts, options=self._options)


class Points(Layout):
    """ A customization of the holoviews.Points class.

        Methods:
            copy(): Create a copy of this object.
            get_parameters(): Get the object's attributes.
            options(**kwargs): Modify the attributes.
    """

    def __init__(self, xpos, ypos, s=5, **kwargs):
        """
        Args:
            xpos (numpy.array): The x-values.
            ypos (numpy.array): The y-values.
            s (int, optional): The point size. Defaults to 5.
        """

        # Assign the attributes.
        super(Points, self).__init__(**kwargs)
        self._xpos = xpos
        self._ypos = ypos
        self._s = s

        # Generate the holoviews.Layout object.
        self._set_layout()


    def get_parameters(self):
        """ Get the attributes of this object.

        Returns:
            dict: A dictionary of attributes.
        """

        out = {}
        out["s"] = self._s
        for key, value in super(Points, self).get_parameters().items():
            out[key] = value
        return out


    def _get_points(self, x, y, label):
        """ Generate the scatter plot.

        Args:
            x (numpy.array): The x-values.
            y (numpy.array): The y-values.
            label (string): The label of the data set.
        """

        out = hv.Points((x, y), label=label).options(
                **self._options).opts(marker="o", size=self._s, **self._opts)
        out = out.options(line_color='black', fill_color=self._color)
        self._layout = out


    def _set_layout(self):
        """ Create the figure. """

        self._get_points(self._xpos, self._ypos, self._title)
        self._layout = self._overlay._apply_layout(self._layout)


    def options(self, **kwargs):
        """ Change the attributes of this object.

        Returns:
            Points: Returns a modified version of itself.
        """

        name_list = ["s"]
        def set_field(prop, key):
            """ Get the attribute value stored in kwargs.

            Args:
                prop (attribute value): The current attribute value.
                key (attribute name): The name of the attribute.

            Returns:
                attribute value: The new attribute value.
            """

            if key in kwargs:
                value = kwargs[key]
                kwargs.pop(key, None)
                return value
            else:
                return prop

        return super(Points, self).options(**kwargs)


    def copy(self, overlay=None):
        """ Create a copy of itself.

        Args:
            overlay (Overlay, optional): The parent overlay class to which the
                                         copy should be assigned. None, if a new
                                         one should be created. Defaults to None.

        Returns:
            Points: A copy of itself.
        """

        return Points(
            self._xpos, self._ypos, **self.get_parameters(),
            overlay=overlay, opts=self._opts, options=self._options)


class Area(Layout):
    """ A customization of the holoviews.Area class.

        Methods:
            copy(): Create a copy of this object.
            get_parameters(): Get the object's attributes.
            options(**kwargs): Modify the attributes.
    """

    def __init__(self, xaxis, data, **kwargs):
        """
        Args:
            xaxis (numpy.array or dict): (A dictionary of) x-values.
            data (numpy.array or dict): (A dictionary of) y-values.
        """

        # Assign the attributes.
        super(Area, self).__init__(**kwargs)
        self._xaxis = xaxis
        self._data = data
        self._options["fill_color"] = self._color
        self._options["alpha"] = self._alpha

        # Generate the holoviews.Layout object.
        self._set_layout()


    def get_parameters(self):
        """ Get the attributes of this object.

        Returns:
            dict: A dictionary of attributes.
        """

        out = {}
        for key, value in super(Area, self).get_parameters().items():
            out[key] = value
        return out


    def _get_area(self, x, y, label, color=None):
        """ Generate the plot.

        Args:
            x (numpy.array): The x-values.
            y (numpy.array): The y-values.
            label (string): The label of the data set.
            color (bokeh.Color, optional): the color of the area. Defaults to None.
        """

        # Plot the area.
        out = hv.Area((x, y), label=label).options(
            **self._options).opts(**self._opts)

        # Set the color.
        if color is not None:
            out = out.options(fill_color=color)

        # Assign the plot to the layout.
        if self._layout is None:
            self._layout = out
        else:
            self._layout *= out

        # Adjust the axes boundaries.
        self._set_limits(x, y)


    def _set_layout(self):
        """ Create the figure. """

        # Initialize the boundaries of the axes.
        self._reset_limits()

        if type(self._data) != dict:
            # Draw a single area.
            yaxis = self._mod_y(self._data)
            self._get_area(self._xaxis, yaxis, self._title)
        elif type(self._xaxis) != dict:
            # The x-axis is the same for all data sets in data.

            # Iterate over all data sets.
            # Define the transparency and color by the index.
            counter = 0
            # TODO: Use enumerate.
            for d in self._data:
                yaxis = self._mod_y(self._data[d])
                self._get_area(
                    self._xaxis, yaxis, d, color=self._set_color(counter))
                counter += 1
        else:
            # Each data set has its own x-axis.

            # Iterate over all data sets.
            # Define the transparency and color by the index.
            counter = 0
            # TODO: Use enumerate.
            for d in self._data:
                yaxis = self._mod_y(self._data[d])
                self._get_area(
                    self._xaxis[d], yaxis, d, color=self._set_color(counter))
                counter += 1

        # Assign the figure.
        self._layout = self._overlay._apply_layout(self._layout)


    def options(self, **kwargs):
        """ Change the attributes of this object.

        Returns:
            Area: Returns a modified version of itself.
        """

        def set_field(prop, key):
            """ Get the attribute value stored in kwargs.

            Args:
                prop (attribute value): The current attribute value.
                key (attribute name): The name of the attribute.

            Returns:
                attribute value: The new attribute value.
            """

            if key in kwargs:
                value = kwargs[key]
                kwargs.pop(key, None)
                return value
            else:
                return prop

        return super(Area, self).options(**kwargs)


    def copy(self, overlay=None):
        """ Create a copy of itself.

        Args:
            overlay (Overlay, optional): The parent overlay class to which the
                                         copy should be assigned. None, if a new
                                         one should be created. Defaults to None.

        Returns:
            Area: A copy of itself.
        """

        return Area(
            self._xaxis, self._data, **self.get_parameters(),
            overlay=overlay, opts=self._opts, options=self._options)