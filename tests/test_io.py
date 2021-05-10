import tempfile
import tables
import numpy as np


def test_writer():
    from project_a5.simulation.detector import Detector
    from project_a5.simulation.particle import CascadeParticle
    from project_a5.io import EventWriter

    rg = np.random.default_rng(42)

    detector = Detector()
    particles = [
        [
            CascadeParticle(energy=10., direction=0.2, x=5., y=5.),
            CascadeParticle(energy=100., direction=0.2, x=5., y=5.),
        ],
        CascadeParticle(energy=10000., direction=0.2, x=70., y=70.),
    ]

    events = [detector.generate_event(p, rg) for p in particles]

    with tempfile.NamedTemporaryFile(prefix='test_io', suffix='.hdf5') as f:
        # write the two events into the file
        with EventWriter(f.name, detector, mode='w') as writer:
            for event in events:
                writer.write_event(event)

        # check if we can read them back using tables
        h5file = tables.open_file(f.name, mode='r')
        event_table = h5file.root.events

        for idx, row in enumerate(event_table.iterrows()):
            assert row['event_id'] == events[idx].event_id
            assert np.all(row['pixels'] == events[idx].pixels)
            assert row['passed_trigger'] == events[idx].passed_trigger

        # make sure we actually read two events
        assert idx == 1

        particle_table = h5file.root.particles

        assert len(particle_table) == 3
        assert particle_table[0]['event_id'] == 0
        assert particle_table[1]['event_id'] == 0
        assert particle_table[2]['event_id'] == 1
