from nt2.containers.fields import FieldsContainer
from nt2.containers.particles import ParticleContainer
from nt2.containers.spectra import SpectraContainer

from nt2.containers.utils import InheritClassDocstring
from nt2.export import makeFramesAndMovie
from nt2.utils import FutureDeprecationWarning

import warnings


@InheritClassDocstring
class Data(FieldsContainer, ParticleContainer, SpectraContainer):
    """
    * * * * Data : FieldsContainer, ParticleContainer, SpectraContainer * * * *

    Master class for holding the whole simulation data.
    Inherits attributes & methods from more specialized classes.

    """

    def __init__(self, **kwargs):
        """
        Kwargs
        ------
        pickle : bool, optional
            Whether to use pickle for reading the data. Default is True.

        greek : bool, optional
            Whether to use Greek letters for the spherical coordinates. Default is False.

        dask_props : dict, optional
            Additional properties for Dask [NOT IMPLEMENTED]. Default is {}.

        """
        if "single_file" in kwargs:
            kwargs.pop("single_file")
            DeprecationWarning()
            warnings.warn(
                """
-= Deprecation Warning =-
The argument `single_file` is no longer needed to be specified, 
as the code derives this from the extension passed in the `path`. 
The argument is ignored now, but will throw an error in future releases.""",
                FutureDeprecationWarning,
            )
        super(Data, self).__init__(**kwargs)
        if "path" not in kwargs:
            raise ValueError('Usage example: data = nt2.Data(path="...", ...)')

    def __repr__(self) -> str:
        help = "Usage: \n"
        help += '  data = Data(path="...", ...)\n'
        help += "  data.fields\n"
        help += "  data.particles\n"
        help += "  data.spectra\n"
        return (
            help
            + "\n"
            + self.print_fields()
            + "\n"
            + self.print_particles()
            + "\n"
            + self.print_spectra()
        )

    def __str__(self) -> str:
        return self.__repr__()

    def __del__(self):
        super().__del__()

    def makeMovie(self, plot, times=None, **kwargs):
        """
        Makes a movie from a plot function

        Parameters
        ----------
        plot : function
            The plot function to use; accepts output timestep indices or timestamps and, optionally,
            the dataset as arguments.

        times : array_like, optional
            Either time indices or timestamps to use for generating the movie. Default is None.
            If None, will use timestamps (not steps) from the fields,
            which might not coincide with values from other quantities.

        **kwargs :
            Additional keyword arguments passed to `ffmpeg`.

        """

        if times is None:
            times = self.fields.t.values
        return makeFramesAndMovie(
            name=self.attrs["simulation.name"],
            data=self,
            plot=plot,
            times=times,
            **kwargs,
        )
