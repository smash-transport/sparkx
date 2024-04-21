#===================================================
#
#    Copyright (c) 2023-2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
#===================================================
    
import numpy as np
import fastjet as fj
import csv
import warnings
from sparkx.Particle import Particle

class JetAnalysis:
    """
    This class analyzes simulation output using the
    `fastjet <https://fastjet.readthedocs.io/en/latest/>`__ python package.
    For further information on the jet finding algorithms please have a look
    at the documentation.

    **Attention**: For a proper jet hole subtraction the `status` of the 
    particles has to be set. This is only given for read in from Jetscape
    output. With Oscar read in this has to be set by hand before doing the 
    jet finding.

    Attributes
    ----------
    hadron_data_: list
        List of hadron data for each event.
    jet_R_: float
        Jet radius parameter.
    jet_eta_range_: tuple
        Minimum and maximum pseudorapidity for jet selection.
    jet_pt_range_: tuple
        Minimum transverse momentum for jet selection and maximum transverse
        momentum to write out the jet.
    jet_data_: list
        List containing the jet data after :code:`read_jet_data` is used.

    Methods
    -------
    create_fastjet_PseudoJets(event_hadrons):
        Convert hadron data to a list of fastjet.PseudoJet objects.
    fill_associated_particles(jet, event):
        Select particles in the jet cone.
    write_jet_output(output_filename, jet, associated_hadrons, new_file=False):
        Write the jet and associated hadron information to a CSV file.
    perform_jet_finding(output_filename):
        Perform the jet analysis for multiple events.
    read_jet_data:
        Read the jet data from a CSV file.
    get_jets:
        Get a list of jets from the jet data.
    get_associated_particles:
        Get a list of associated particles for all jets.

    Examples
    --------
    First import your hadron data and maybe perform some cuts:

    .. highlight:: python
    .. code-block:: python
        :linenos:

        >>> from sparkx.Jetscape import Jetscape
        >>> from sparkx.JetAnalysis import JetAnalysis
        >>>
        >>> JETSCAPE_FILE_PATH = [Jetscape_directory]/particle_lists.dat
        >>>
        >>> # Jetscape object containing all events
        >>> hadron_data_set = Jetscape(JETSCAPE_FILE_PATH)
        >>> hadron_data = hadron_data_set.charged_particles().particle_objects_list()

    Then you can perform the jet analysis:

    .. highlight:: python
    .. code-block:: python
        :linenos:

        >>> JET_ANALYSIS_OUTPUT_PATH = [Jetscape_directory]/jet_analysis.dat
        >>>
        >>> # Create an instance of the JetAnalysis class
        >>> jet_analysis = JetAnalysis()
        >>>
        >>> # Perform the jet analysis
        >>> jet_analysis.perform_jet_finding(hadron_data, jet_R=0.4, jet_eta_range=(-2., 2.), jet_pt_range=(10., None), output_filename=JET_ANALYSIS_OUTPUT_PATH)

    If you want to analyze the jets further, you have to read in the jet data
    from the file like:

    .. highlight:: python
    .. code-block:: python
        :linenos:

        >>> # Create an instance of the JetAnalysis class
        >>> jet_analysis = JetAnalysis()
        >>>
        >>> # Read the jets from file
        >>> jet_analysis.read_jet_data(JET_ANALYSIS_OUTPUT_PATH)
        >>>
        >>> # list of the jets (all lines with index 0 in first column)
        >>> jets = jet_analysis.get_jets()
        >>> # list of the associated particles for all jets (associated hadrons for each jet have a sub-list)
        >>> assoc_hadrons = jet_analysis.get_associated_particles()

    Notes
    -----
    The columns of the output file include the following quantities:

    .. list-table::
        :header-rows: 1
        :widths: 25 75

        * - Quantity
          - Description
        * - index
          - Index of the particle in the jet. Index 0 is always the jet itself, all higher values are the associated particles of the jet.
        * - :math:`p_{\\mathrm{T}}`
          - Transverse momentum of the jet / associated particle.
        * - :math:`\\eta`
          - Pseudorapidity of the jet / associated particle.
        * - :math:`\\varphi`
          - Azimuth of the jet / associated particle.
        * - status flag
          - Always set to 10.
        * - jet pid
          - Always set to 10.
        * - energy
          - Energy of the jet / associated particle.
        * - event index
          - Index of the event in the input hadron data.

    .. |br| raw:: html

           <br />
    """
    def __init__(self):
        self.hadron_data_ = None
        self.jet_R_ = None
        self.jet_eta_range_ = None
        self.jet_pt_range_ = None
        self.jet_data_ = None

    def __initialize_and_check_parameters(self, hadron_data, jet_R, jet_eta_range, 
                                          jet_pt_range):
        """
        Initialize and check the parameters for jet analysis.

        Parameters
        ----------
        hadron_data: list
            List of hadron data for each event.
            Use the for example the Jetscape class and generate 
            a :code:`particle_objects_list`.
        jet_R: float
            Jet radius parameter.
        jet_eta_range: tuple
            Minimum and maximum pseudorapidity for jet selection.
            :code:`None` values are allowed and are exchanged
            by :math:`-\\infty` or :math:`+\\infty` automatically.
        jet_pt_range: tuple
            Minimum transverse momentum for jet finding algorithm and maximum
            transverse momentum to write out the jet to a file.
            Values can be :code:`None`, then the lower bound is set to zero
            and the upper one to :math:`+\\infty`.
        """
        self.hadron_data_ = hadron_data

        # check jet radius
        if jet_R <= 0.:
            raise ValueError("jet_R must be larger than 0")
        self.jet_R_ = jet_R

        # check jet eta range
        if not isinstance(jet_eta_range, tuple):
            raise TypeError("jet_eta_range is not a tuple. " +\
                            "It must contain either values or None.")
        if len(jet_eta_range) != 2:
            raise ValueError("jet_eta_range must contain exactly two values.")

        lower_cut = float('-inf') if jet_eta_range[0] is None else jet_eta_range[0]
        upper_cut = float('inf') if jet_eta_range[1] is None else jet_eta_range[1]
        if lower_cut < upper_cut:
            self.jet_eta_range_ = (lower_cut, upper_cut)
        else:
            self.jet_eta_range_ = (upper_cut, lower_cut)
            warnings.warn("The lower jet eta cut value is larger than the " +\
                          "upper one. They are interchanged automatically.")

        # check the jet pt range
        if not isinstance(jet_pt_range, tuple):
            raise TypeError("jet_pt_range is not a tuple. " +\
                            "It must contain either values or None.")
        if len(jet_pt_range) != 2:
            raise ValueError("jet_pt_range must contain exactly two values.")
        if any(pt is not None and pt < 0 for pt in jet_pt_range):
            raise ValueError("All values in jet_pt_range must be non-negative.")

        lower_cut = 0. if jet_pt_range[0] is None else jet_pt_range[0]
        upper_cut = float('inf') if jet_pt_range[1] is None else jet_pt_range[1]
        if lower_cut < upper_cut:
            self.jet_pt_range_ = (lower_cut, upper_cut)
        else:
            self.jet_pt_range_ = (upper_cut, lower_cut)
            warnings.warn("The lower jet pt cut value is larger than the " +\
                          "upper one. They are interchanged automatically.")

    def create_fastjet_PseudoJets(self, event_hadrons):
        """
        Convert hadron data to a list of fastjet.PseudoJet objects.

        Parameters
        ----------
        event_hadrons: list
            List of Particle objects representing hadrons.

        Returns
        -------
        list
            List of fastjet.PseudoJet objects.
        """
        event_list_PseudoJets = [
            fj.PseudoJet(hadron.px, hadron.py, hadron.pz, hadron.E)
            for hadron in event_hadrons
        ]
        return event_list_PseudoJets

    def fill_associated_particles(self,jet,event,status_selection,only_charged):
        """
        Select particles in the jet cone.

        Parameters
        ----------
        jet: fastjet.PseudoJet
            The jet to check for associated particles.
        event: int
            The event index.
        status_selection: str
            The selection status to find the associated particles. This can be 
            'negative' or 'positive' hadrons status.
        only_charged: bool
            Select only charged particles as associated particles.

        Returns
        -------
        list
            List of associated particles in the jet cone.
        """
        associated_hadrons = []
        for hadron in self.hadron_data_[event]:
            if np.isnan(hadron.status):
                raise ValueError("Hadron status not set")
            if (status_selection == 'negative' and hadron.status >= 0) or \
                (status_selection == 'positive' and hadron.status < 0) or \
                (only_charged and hadron.charge == 0):
                continue

            fastjet_particle = fj.PseudoJet(hadron.px,hadron.py,\
                                            hadron.pz,hadron.E)
            delta_eta = fastjet_particle.eta() - jet.eta()
            delta_phi = fastjet_particle.delta_phi_to(jet)
            delta_r = np.sqrt(delta_eta**2. + delta_phi**2.)

            if delta_r < self.jet_R_:
                associated_hadrons.append(hadron)
        return associated_hadrons

    def jet_hole_subtraction(self,jet,holes):
        """
        Subtract energy-momentum contributions from holes in the jet.

        Parameters
        ----------
        jet : fastjet.PseudoJet
            The jet object from which energy-momentum contributions will be 
            subtracted.
        holes : list
            List of hadrons representing holes contributing to the jet's 
            energy-momentum.
    
        Returns
        -------
        fastjet.PseudoJet
            The jet object after subtracting energy-momentum contributions 
            from holes.
        """
        E = 0.0
        px = 0.0
        py = 0.0
        pz = 0.0

        for hole in holes:
            E += hole.E
            px += hole.px
            py += hole.py
            pz += hole.pz

        # subtraction of hole energy-momentum
        jet_E = jet.e() - E
        jet_px = jet.px() - px
        jet_py = jet.py() - py
        jet_pz = jet.pz() - pz
        jet.reset(jet_px, jet_py, jet_pz, jet_E)
        return jet

    def write_jet_output(self,output_filename,jet,associated_hadrons,event,
                         new_file=False):
        """
        Write the jet and associated hadron information to a CSV file.

        Parameters
        ----------
        output_filename: str
            Filename for the jet output.
        jet: fastjet.PseudoJet
            The jet object.
        associated_hadrons: list
            List of associated hadrons.
        event: int
            Event index.
        new_file: bool, optional
            Whether to create a new file or append to an existing file.
            Default is False.

        Returns
        -------
        bool
            False if successful.
        """
        # jet data from reconstruction
        jet_status = 10
        jet_pid = 10
        output_list = []

        if jet.perp() < self.jet_pt_range_[1]:
            output_list = [[0,jet.perp(),jet.eta(),jet.phi(),jet_status,\
                            jet_pid,jet.e(),event]]

            # associated hadrons
            for i, associated in enumerate(associated_hadrons, start=1):
                pseudo_jet = fj.PseudoJet(associated.px,associated.py,\
                                          associated.pz,associated.E)
                output = [i,pseudo_jet.perp(),pseudo_jet.eta(),\
                          pseudo_jet.phi(),associated.status,\
                            associated.pdg,associated.E,event]
                output_list.append(output)

        mode = 'a' if not new_file else 'w'

        with open(output_filename, mode, newline='') as f:
            writer = csv.writer(f)
            writer.writerows(output_list)

        return False

    def perform_jet_finding(self, hadron_data, jet_R, jet_eta_range, jet_pt_range, 
                            output_filename, assoc_only_charged=True, 
                            jet_algorithm=fj.antikt_algorithm):
        """
        Perform the jet analysis for multiple events. The function generates a
        file containing the jets consisting of a leading particle and associated
        hadrons in the jet cone.

        Parameters
        ----------
        hadron_data: list
            List of hadron data for each event.
            Use the for example the Jetscape class and generate 
            a :code:`particle_objects_list`.
        jet_R: float
            Jet radius parameter.
        jet_eta_range: tuple
            Minimum and maximum pseudorapidity for jet selection.
            :code:`None` values are allowed and are exchanged
            by :math:`-\\infty` or :math:`+\\infty` automatically.
        jet_pt_range: tuple
            Minimum transverse momentum for jet finding algorithm and maximum
            transverse momentum to write out the jet to a file.
            Values can be :code:`None`, then the lower bound is set to zero
            and the upper one to :math:`+\\infty`.
        output_filename: str
            Filename for the jet output.
        assoc_only_charged: bool
            Selects if the associated particles are only charged ones. 
            Default is True.
        jet_algorithm: fastjet.JetAlgorithm, optional
            Jet algorithm for jet finding. 
            Default is :code:`fastjet.antikt_algorithm`.

        Notes
        -----
        The standard recombination scheme :code:`E_scheme` of the fastjet 
        package is fixed here.

        """
        self.__initialize_and_check_parameters(hadron_data, jet_R, jet_eta_range, jet_pt_range)
        for event, hadron_data_event in enumerate(self.hadron_data_):
            new_file = False
            event_PseudoJets = self.create_fastjet_PseudoJets(hadron_data_event)
            if jet_algorithm == fj.ee_genkt_algorithm or jet_algorithm == fj.genkt_algorithm:
                jet_definition = fj.JetDefinition(jet_algorithm, self.jet_R_, -1.0)
            else:
                jet_definition = fj.JetDefinition(jet_algorithm, self.jet_R_)
            jet_selector = fj.SelectorEtaRange(self.jet_eta_range_[0],self.jet_eta_range_[1])

            if event == 0:
                print("jet definition is:", jet_definition)
                print("jet selector is:", jet_selector)
                # create a new file for the first event in the dataset
                new_file = True

            # perform the jet finding algorithm
            cluster = fj.ClusterSequence(event_PseudoJets, jet_definition)
            jets = fj.sorted_by_pt(cluster.inclusive_jets(self.jet_pt_range_[0]))
            jets = jet_selector(jets)

            # get the associated particles in the jet cone
            for jet in jets:
                holes_in_jet = self.fill_associated_particles(jet, event, 
                                        status_selection='negative',
                                        only_charged=assoc_only_charged)
                associated_particles = self.fill_associated_particles(jet, event, 
                                                status_selection='positive',
                                                only_charged=assoc_only_charged)
                jet = self.jet_hole_subtraction(jet, holes_in_jet)
                new_file = self.write_jet_output(output_filename, jet,\
                                                associated_particles,\
                                                event,new_file)

    def read_jet_data(self, input_filename):
        """
        Read the jet data from a CSV file and store it in the
        :code:`JetAnalysis` object.

        Parameters
        ----------
        input_filename: str
            Filename of the CSV file containing the jet data.
        """
        jet_data = []
        current_jet = []
        with open(input_filename, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                jet_index = int(row[0])
                if jet_index == 0 and current_jet:
                    jet_data.append(current_jet)
                    current_jet = []
                current_jet.append([int(row[0]), float(row[1]), float(row[2]),\
                                     float(row[3]), int(row[4]), int(row[5]),\
                                          float(row[6]), int(row[7])])
            if current_jet:
                jet_data.append(current_jet)
        self.jet_data_ = jet_data

    def get_jets(self):
        """
        Get a list of jets from the jet data.

        Returns
        -------
        list
            List of jets. Contains all data of the jet output file rows in each
            element of the list (:code:`[jet][column]`).
        """
        return [jet[0] for jet in self.jet_data_]

    def get_associated_particles(self):
        """
        Get a list of associated particles for all jets.

        Returns
        -------
        list
            List of associated particles for each jet in each element
            (:code:`[jet][associated_particle][column]`).
        """
        associated_particles_list = []
        for jet in self.jet_data_:
            associated_particles = jet[1:]
            associated_particles_list.append(associated_particles)
        return associated_particles_list
