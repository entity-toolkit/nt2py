from nt2.containers.fields import FieldsContainer
from nt2.containers.particles import ParticleContainer
from nt2.containers.spectra import SpectraContainer


class Data(FieldsContainer, ParticleContainer, SpectraContainer):
    """
    A class to load Entity data and store it as a lazily loaded xarray Dataset.
    """

    def __init__(self, **kwargs):
        super(Data, self).__init__(**kwargs)

    def __repr__(self) -> str:
        help = "Usage: \n"
        help += "  data = Data(path, ...)\n"
        help += "  data.fields\n"
        help += "  data.particles\n"
        help += "  data.spectra\n"
        return (
            help
            + "\n"
            + self.print_container()
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
        self.client.close()
        super().__del__()
