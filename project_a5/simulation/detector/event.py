import numpy as np


class Event():

    """A class that contains information of an event.

    Attributes
    ----------
    event_id : int, optional
        The event identification number.
    name : str
        The name of the event.
    particles : list of particles
        A list of particles that are contained in the event.
    passed_trigger : bool
        Indicates whether the event passed the detector trigger.
    pixels : array_like
            The pixel response of the event.
    """

    def __init__(
            self,
            pixels,
            particles,
            passed_trigger,
            event_id=0,
            name='Event'):
        """Create an event.

        Parameters
        ----------
        pixels : array_like
            The pixel response of the event.
        particles : list of particles
            A list of particles that are contained in the event.
        passed_trigger : bool
            If True, event passes detector trigger.
            If False, event does not pass detector trigger.
        event_id : int, optional
            An optional event identification number.
        name : str, optional
            An optional name of the event.
        """

        # assign values
        self.pixels = pixels
        self.particles = particles
        self.passed_trigger = passed_trigger
        self.event_id = event_id
        self.name = name
