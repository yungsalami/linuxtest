import tables
import numpy as np
from pandas import DataFrame


class EventWriter:
    def __init__(self, path, detector, mode='a'):

        class Event(tables.IsDescription):
            event_id = tables.UInt32Col()
            pixels = tables.Float64Col(shape=detector.event_shape)
            passed_trigger = tables.BoolCol()

        class Particle(tables.IsDescription):
            event_id = tables.UInt32Col()
            energy = tables.Float64Col()
            direction = tables.Float64Col()
            x = tables.Float64Col()
            y = tables.Float64Col()
            type = tables.StringCol(10)

        self.event_schema = Event
        self.particle_schema = Particle
        self.file = tables.open_file(path, mode=mode)

        if 'events' not in self.file.root:
            self.event_table = self.file.create_table(
                '/', 'events', Event
            )
        else:
            self.event_table = self.file.root.events

        if 'particles' not in self.file.root:
            self.particle_table = self.file.create_table(
                '/', 'particles', Particle
            )
        else:
            self.particle_table = self.file.root.particles

    def write_event(self, event):
        row = self.event_table.row
        for col in self.event_table.colnames:
            row[col] = getattr(event, col)
        row.append()

        for particle in event.particles:
            row = self.particle_table.row
            row['event_id'] = event.event_id
            row['energy'] = particle.energy
            row['direction'] = particle.direction
            row['type'] = particle.__class__.__name__[:10]

            # currently, only cascades have x and y
            for attr in ('x', 'y'):
                row[attr] = getattr(particle, attr, np.nan)

            row.append()

    def __enter__(self):
        return self

    def __exit__(self, exc_value, exc_type, traceback):
        self.file.close()


def read_particles(path):
    """Read back the particles of the events written with `EventWriter`.

    A simple tables --> pandas.DataFrame wrapper.

    Parameters
    ----------
    path : str or pathlib.Path

    Returns
    -------
    pandas.DataFrame
    """
    with tables.open_file(path, mode="r") as f:
        particles = DataFrame(f.root.particles[:])

    return particles
