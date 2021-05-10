import numpy as np


class FeatureGenerator():
    """This class generates features for an event

    Based on the values measured in the detector, arbitrary features
    can be generated. These can be used for machine learning exercises
    such as classifying the particle type or estimating energy and origin.
    """

    def __init__(self, detector):
        """Sets up the feature generator

        Parameters
        ----------
        detector : Detector object
            A detector object for which the feature generator will be set up.
        """
        self.detector = detector

    def analyse(self, event):
        """Reconstruct an event measured in the assigned detector.

        Parameters
        ----------
        event : Event
            The event which we want to reconstruct.

        Returns
        -------
        Features: Dict
            A dictionary of arbitrary features.
            Values are to be scalar to be fed into a random forest
        """

        # ---------
        # Exercise:
        # ---------
        # Feature Generation (exercises/feature_generation.py)
        #
        # ---------------------------------------------------------------------
        # Replace Code here
        # ---------------------------------------------------------------------

        # Dummy solution to pass unit tests (this solution is not correct)
        # You can access the event and detector here.
        features = {
                'mean_pixel_value': np.mean(event.pixels),
                'detector_length_x': self.detector.num_pixels_x
        }


        return features
