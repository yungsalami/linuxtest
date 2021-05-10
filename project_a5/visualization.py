'''
Plotting functions using matplotlib
'''
import matplotlib.pyplot as plt
import numpy as np

from .simulation.detector.event import Event


class EventDisplay:
    ''' Matplotlib Event Display'''

    def __init__(self, detector, event=None, ax=None, **kwargs):
        '''
        Create an event display for a detector.

        Parameters
        ----------
        detector: Detector
            The detector for which to create the display
        event: ndarray of correct shape or None
            If None, an empty event will be used to build the initial display
        ax: matplotlib.axes.Axes or None
            if None, the current axes will be used, if there is no current
            axes, a new one will be created.

        **kwargs are passed to ax.pcolormesh
        '''
        self.ax = ax or plt.gca()
        self.fig = self.ax.figure
        self.detector = detector
        self.colorbar = None
        if event is None:
            self.event = Event(np.ones(detector.event_shape), [], True)
        else:
            self.event = event

        x = np.linspace(0, detector.x_extend, detector.num_pixels_x + 1)
        y = np.linspace(0, detector.y_extend, detector.num_pixels_y + 1)

        self.pixels = self.ax.pcolormesh(
            x, y, self.event.pixels.T, **kwargs
        )
        self.ax.set_aspect(1)

    def add_colorbar(self, **kwargs):
        """
        add a colorbar to the plot
        kwargs are passed to `figure.colorbar(self.pixels, **kwargs)`
        See matplotlib documentation for the supported kwargs:
        http://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure.colorbar
        """
        if self.colorbar is not None:
            raise ValueError(
                'There is already a colorbar attached to this CameraDisplay'
            )
        else:
            ax = kwargs.pop('ax', self.ax)
            self.colorbar = self.fig.colorbar(self.pixels, ax=ax, **kwargs)

        self.update()

    def update(self, force=False):
        """ redraw the display now """
        self.ax.figure.canvas.draw()
        if self.colorbar is not None:
            if force is True:
                self.colorbar.update_bruteforce(self.pixels)
            else:
                self.colorbar.update_normal(self.pixels)
            self.colorbar.draw_all()

    def add_particle(self, particle, length=10, width=0.2, head_width=2, **kwargs):
        dx = length * np.cos(particle.direction)
        dy = length * np.sin(particle.direction)

        self.ax.plot(
            particle.x,
            particle.y,
            marker='.',
            color=kwargs.get('color', 'C0'),
        )
        self.ax.arrow(
            particle.x,
            particle.y,
            dx,
            dy,
            color=kwargs.pop('color', 'C0'),
            width=width,
            head_width=head_width,
            length_includes_head=True,
            **kwargs
        )

    def clear_particles(self):
        for l in self.ax.lines[:]:
            l.remove()
        for p in self.ax.artists[:]:
            p.remove()

    def set_event(self, event):
        pixels = np.asanyarray(event.pixels)
        if pixels.shape != self.detector.event_shape:
            raise ValueError(
                f'Event has wrong shape {pixels.shape}'
                f', should be {self.detector.event_shape}'
            )

        self.pixels.set_array(np.ma.masked_invalid(pixels.T).ravel())
        self.pixels.autoscale()
        self.pixels.changed()
        self.update()
