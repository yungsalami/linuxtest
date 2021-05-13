import numpy as np
import numpy.random


class BaseGenerator(numpy.random.Generator):
    def __init__(self, seed=0):
        super().__init__(numpy.random.PCG64(seed))


class LCG:
    '''A linear congruential generator'''

    def __init__(self, seed=0, a=1664525, c=1013904223, m=2**32):
        '''store seed and LCG properties'''
        self.a = a
        self.c = c
        self.m = m
        # Value from previous iteration
        self.state = seed

    def advance(self):
        '''
        advance the state of this random generator by one step

        Blatt 3, Aufgabe 1
        '''

        # Fügen Sie hier Ihren Code ein, um den LCG korrekt zu implementieren

    def random_raw(self, size=None):
        '''
        Draw raw LCG random numbers, meaning no transformation is happening
        '''
        # if size is not given, just return a single number
        if size is None:
            self.advance()
            return self.state

        numbers = np.empty(size, dtype='uint64')
        # get a 1d-reference to the nd-array so we can fill it in
        # simple loop
        flat = numbers.flat

        for i in range(numbers.size):
            self.advance()
            flat[i] = self.state

        return numbers

    def uniform(self, low=0, high=1, size=None):
        '''
        Draw uniforn random numbers by converting them from the raw numbers'''
        raw = self.random_raw(size=size)

        # Fügen Sie hier ihren Code ein, um die rohen Zufallszahlen
        # zu kontinuierlich gleichverteilten Zufallszahlen zwischen
        # low und high zu transformation
        result = raw
        return result


class Generator(BaseGenerator):
    '''
    Generate random numbers from different distribtions

    self.uniform(size=size) creates standard uniform random numbers.
    '''

    def exponential(self, tau, size=None):
        '''
        Draw exponentially distributed random numbers.

        Blatt 3, Aufgabe 2a)
        '''
        # So können Sie ein array mit shape=size
        # mit standard gleichverteilten Zufallszahlen erzeugen
        u = self.uniform(size=size)

        # Fügen Sie hier den Code ein um Zufallszahlen aus der
        # angegebenen Verteilung zu erzeugen

        # dummy, so the code works. Can be removed / replaced
        values = np.zeros(size)

        return values

    def power(self, n, x_min, x_max, size=None):
        '''
        Draw random numbers from a power law distribution
        with index n between x_min and x_max

        Blatt 3, Aufgabe 2b)
        '''
        # Fügen Sie hier den Code ein um Zufallszahlen aus der
        # angegebenen Verteilung zu erzeugen

        # dummy, so the code works. Can be removed / replaced
        values = np.zeros(size)


        return values

    def cauchy(self, size=None):
        '''
        Draw random numbers from a power law distribution
        with index n between x_min and x_max

        Blatt 3, Aufgabe 2c)
        '''
        # Fügen Sie hier den Code ein um Zufallszahlen aus der
        # angegebenen Verteilung zu erzeugen

        # dummy, so the code works. Can be removed / replaced
        values = np.zeros(size)


        return values

    def standard_normal(self, size=None):
        '''
        Override standard normal with using the Marsaglia polar method

        Blatt 5, Aufgabe 11b)
        '''
        # Fügen Sie hier den Code ein um Zufallszahlen aus der
        # angegebenen Verteilung zu erzeugen

        # dummy, so the code works. Can be removed / replaced
        values = np.zeros(size)


        return values

    def normal(self, loc=0, scale=1, size=None):
        '''
        Scale and shift standard normal values

        Blatt 5, Aufgabe 11b)
        '''
        # Fügen Sie hier den Code ein um Zufallszahlen aus der
        # angegebenen Verteilung zu erzeugen

        # dummy, so the code works. Can be removed / replaced
        values = np.zeros(size)

        return values