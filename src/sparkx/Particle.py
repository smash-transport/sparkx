import numpy as np
import math
from particle import PDGID

class Particle:
    """Defines a particle object.

    The member variables of the Particle class are the quantities in the
    OSCAR2013/OSCAR2013Extended or JETSCAPE hadron output. If they are not set,
    they stay None to throw an error if one tries to access a non existing
    quantity.

    Attributes
    ----------
    t_ : float
        The time of the particle.
    x_ : float
        The x coordinate of the position.
    y_ : float
        The y coordinate of the position.
    z_ : float
        The z coordinate of the position.
    mass_ : float
        The mass of the particle.
    E_ : float
        The energy of the particle.
    px_ : float
        The x component of the momentum.
    py_ : float
        The y component of the momentum.
    pz_ : float
        The z component of the momentum.
    pdg_ : int
        The PDG code of the particle.
    ID_ : int
        The ID of the particle (unique label of each particle in an event).
    charge_ : int
        Electric charge of the particle.
    ncoll_ : int
        Number of collisions undergone by the particle.
    form_time_ : double
        Formation time of the particle.
    xsecfac_ : double
        Scaling factor of the cross section.
    proc_id_origin_ : int
        ID for the process the particle stems from.
    proc_type_origin_ : int
        Type information for the process the particle stems from.
    t_last_coll_ : double
        Time of the last collision.
    pdg_mother1_ : int
        PDG code of the mother particle 1.
    pdg_mother2_ : int
        PDG code of the mother particle 2.
    status_ : int
        Status code of the particle.
    baryon_number_ : int
        Baryon number of the particle.
    strangeness_ : int
        Strangeness quantum number of the particle.
    weight_ : float
        Weight of the particle.

    Methods
    -------
    t:
        Get/set t
    x:
        Get/set x
    y:
        Get/set y
    z:
        Get/set z
    mass:
        Get/set mass
    E:
        Get/set E
    px:
        Get/set px
    py:
        Get/set py
    pz:
        Get/set pz
    pdg:
        Get/set pdg
    ID:
        Get/set ID
    charge:
        Get/set charge
    ncoll:
        Get/set ncoll
    form_time:
        Get/set form_time
    xsecfac:
        Get/set xsecfactor
    proc_id_origin:
        Get/set proc_id_origin
    proc_type_origin:
        Get/set proc_type_origin
    t_last_coll:
        Get/set t_last_coll
    pdg_mother1:
        Get/set pdg_mother1
    pdg_mother2:
        Get/set pdg_mother2
    status:
        Get/set status
    baryon_number:
        Get/set baryon_number
    strangeness:
        Get/set strangeness
    print_particle:
        Print the particle as CSV to terminal
    set_quantities_OSCAR2013:
        Set particle properties OSCAR2013
    set_quantities_OSCAR2013Extended:
        Set particle properties OSCAR2013Extended
    set_quantities_JETSCAPE:
        Set particle properties JETSCAPE
    angular_momentum:
        Compute angular momentum
    momentum_rapidity_Y:
        Compute momentum rapidity
    p_abs:
        Compute absolute momentum
    pt_abs:
        Compute absolute value of transverse momentum
    phi:
        Compute azimuthal angle
    theta:
        Compute polar angle
    pseudorapidity:
        Compute pseudorapidity
    spatial_rapidity:
        Compute spatial rapidity
    proper_time:
        Compute proper time
    compute_mass_from_energy_momentum:
        Compute mass from energy momentum relation
    compute_charge_from_pdg:
        Compute charge from PDG code
    is_meson:
        Is the particle a meson?
    is_baryon:
        Is the particle a baryon?
    is_hadron:
        Is the particle a hadron?
    is_strange:
        Is the particle a strange particle?
    is_heavy_flavor:
        Is the particle a heavy flavor particle?
    weight:
        What is the weight of the particle?

    Notes
    -----
    To set a value of one of the attributes by hand, please use the
    corresponding getter methods with parentheses including the value for the
    attribute of the particle.

    Examples
    --------
    To use the particle class a particle has to be created and the attributes
    can be set or obtained with the corresponding functions.

    .. highlight:: python
    .. code-block:: python
        :linenos:

        >>> from sparkx.Particle import Particle
        >>>
        >>> particle = Particle()
        >>> particle.t = 1.0
        >>> print(particle.t)
        1.0

    """
    def __init__(self):
        self.t_ = None
        self.x_ = None
        self.y_ = None
        self.z_ = None
        self.mass_ = None
        self.E_ = None
        self.px_ = None
        self.py_ = None
        self.pz_ = None
        self.pdg_ = None
        self.ID_ = None
        self.charge_ = None
        self.ncoll_ = None
        self.form_time_ = None
        self.xsecfac_ = None
        self.proc_id_origin_ = None
        self.proc_type_origin_ = None
        self.t_last_coll_ = None
        self.pdg_mother1_ = None
        self.pdg_mother2_ = None
        self.status_ = None
        self.baryon_number_ = None
        self.strangeness_ = None
        self.weight_ = None

    @property
    def t(self):
        """Get or set the time of the particle.

        Returns
        -------
        t_ : float

        Raises
        ------
        ValueError
            if time is not set
        """
        if self.t_ == None:
            raise ValueError("t not set")
        else:
            return self.t_

    @t.setter
    def t(self,value):
        self.t_ = value

    @property
    def x(self):
        """Get or set the x-position of the particle.

        Returns
        -------
        x_ : float

        Raises
        ------
        ValueError
            if x is not set
        """
        if self.x_ == None:
            raise ValueError("x not set")
        else:
            return self.x_

    @x.setter
    def x(self,value):
        self.x_ = value

    @property
    def y(self):
        """Get or set the y-position of the particle.

        Returns
        -------
        y_ : float

        Raises
        ------
        ValueError
            if y is not set
        """
        if self.y_ == None:
            raise ValueError("y not set")
        else:
            return self.y_

    @y.setter
    def y(self,value):
        self.y_ = value

    @property
    def z(self):
        """Get or set the z-position of the particle.

        Returns
        -------
        z_ : float

        Raises
        ------
        ValueError
            if z is not set
        """
        if self.z_ == None:
            raise ValueError("z not set")
        else:
            return self.z_

    @z.setter
    def z(self,value):
        self.z_ = value

    @property
    def mass(self):
        """Get or set the mass of the particle.

        Returns
        -------
        mass_ : float

        Raises
        ------
        ValueError
            if mass is not set
        """
        if self.mass_ == None:
            raise ValueError("mass not set")
        else:
            return self.mass_

    @mass.setter
    def mass(self,value):
        self.mass_ = value

    @property
    def E(self):
        """Get or set the energy of the particle.

        Returns
        -------
        E_ : float

        Raises
        ------
        ValueError
            if E is not set
        """
        if self.E_ == None:
            raise ValueError("E not set")
        else:
            return self.E_

    @E.setter
    def E(self,value):
        self.E_ = value

    @property
    def px(self):
        """Get or set the momentum x-component of the particle.

        Returns
        -------
        px_ : float

        Raises
        ------
        ValueError
            if px is not set
        """
        if self.px_ == None:
            raise ValueError("px not set")
        else:
            return self.px_

    @px.setter
    def px(self,value):
        self.px_ = value

    @property
    def py(self):
        """Get or set the momentum y-component of the particle.

        Returns
        -------
        py_ : float

        Raises
        ------
        ValueError
            if py is not set
        """
        if self.py_ == None:
            raise ValueError("py not set")
        else:
            return self.py_

    @py.setter
    def py(self,value):
        self.py_ = value

    @property
    def pz(self):
        """Get or set the momentum z-component of the particle.

        Returns
        -------
        pz_ : float

        Raises
        ------
        ValueError
            if pz is not set
        """
        if self.pz_ == None:
            raise ValueError("pz not set")
        else:
            return self.pz_

    @pz.setter
    def pz(self,value):
        self.pz_ = value

    @property
    def pdg(self):
        """Get or set the PDG code of the particle.

        Returns
        -------
        pdg_ : int

        Raises
        ------
        ValueError
            if pdg is not set
        """
        if self.pdg_ == None:
            raise ValueError("pdg not set")
        else:
            return self.pdg_

    @pdg.setter
    def pdg(self,value):
        self.pdg_ = value

    @property
    def ID(self):
        """Get or set the ID of the particle.

        This is a unique number in SMASH.

        Returns
        -------
        ID_ : int

        Raises
        ------
        ValueError
            if ID is not set
        """
        if self.ID_ == None:
            raise ValueError("ID not set")
        else:
            return self.ID_

    @ID.setter
    def ID(self,value):
        self.ID_ = value

    @property
    def charge(self):
        """Get or set the electrical charge of the particle.

        Returns
        -------
        charge_ : int

        Raises
        ------
        ValueError
            if charge is not set
        """
        if self.charge_ == None:
            raise ValueError("charge not set")
        else:
            return self.charge_

    @charge.setter
    def charge(self,value):
        self.charge_ = value

    @property
    def ncoll(self):
        """Get or set the number of collisions of the particle.

        Returns
        -------
        ncoll_ : int

        Raises
        ------
        ValueError
            if ncoll is not set
        """
        if self.ncoll_ == None:
            raise ValueError("ncoll not set")
        else:
            return self.ncoll_

    @ncoll.setter
    def ncoll(self,value):
        self.ncoll_ = value

    @property
    def form_time(self):
        """Get or set the formation time of the particle.

        Returns
        -------
        form_time_ : float

        Raises
        ------
        ValueError
            if form_time is not set
        """
        if self.form_time_ == None:
            raise ValueError("form_time not set")
        else:
            return self.form_time_

    @form_time.setter
    def form_time(self,value):
        self.form_time_ = value

    @property
    def xsecfac(self):
        """Get or set the crosssection scaling factor of the particle.

        Returns
        -------
        xsecfac_ : float

        Raises
        ------
        ValueError
            if xsecfac is not set
        """
        if self.xsecfac_ == None:
            raise ValueError("xsecfac not set")
        else:
            return self.xsecfac_

    @xsecfac.setter
    def xsecfac(self,value):
        self.xsecfac_ = value

    @property
    def proc_id_origin(self):
        """Get or set the process ID of the particle's origin.

        Returns
        -------
        proc_id_origin_ : int

        Raises
        ------
        ValueError
            if proc_id_origin is not set
        """
        if self.proc_id_origin_ == None:
            raise ValueError("proc_id_origin not set")
        else:
            return self.proc_id_origin_

    @proc_id_origin.setter
    def proc_id_origin(self,value):
        self.proc_id_origin_ = value

    @property
    def proc_type_origin(self):
        """Get or set the process type of the particle's origin.

        Returns
        -------
        proc_type_origin_ : int

        Raises
        ------
        ValueError
            if proc_type_origin is not set
        """
        if self.proc_type_origin_ == None:
            raise ValueError("proc_type_origin not set")
        else:
            return self.proc_type_origin_

    @proc_type_origin.setter
    def proc_type_origin(self,value):
        self.proc_type_origin_ = value

    @property
    def t_last_coll(self):
        """Get or set the last time of a collision of the particle.

        Returns
        -------
        t_last_coll_ : float

        Raises
        ------
        ValueError
            if t_last_coll is not set
        """
        if self.t_last_coll_ == None:
            raise ValueError("t_last_coll not set")
        else:
            return self.t_last_coll_

    @t_last_coll.setter
    def t_last_coll(self,value):
        self.t_last_coll_ = value

    @property
    def pdg_mother1(self):
        """Get the PDG code of the first mother particle.

        Returns
        -------
        pdg_mother1_ : int

        Raises
        ------
        ValueError
            if pdg_mother1 is not set
        """
        if self.pdg_mother1_ == None:
            raise ValueError("pdg_mother1 not set")
        else:
            return self.pdg_mother1_

    @pdg_mother1.setter
    def pdg_mother1(self,value):
        self.pdg_mother1_ = value

    @property
    def pdg_mother2(self):
        """Get the PDG code of the second mother particle.

        Returns
        -------
        pdg_mother2_ : int

        Raises
        ------
        ValueError
            if pdg_mother2 is not set
        """
        if self.pdg_mother2_ == None:
            raise ValueError("pdg_mother2 not set")
        else:
            return self.pdg_mother2_

    @pdg_mother2.setter
    def pdg_mother2(self,value):
        self.pdg_mother2_ = value

    @property
    def status(self):
        """
        Get the hadron status (stores information on the module origin of
        a JETSCAPE hadron).

        Returns
        -------
        status_ : int

        Raises
        ------
        ValueError
            if status is not set
        """
        if self.status_ == None:
            raise ValueError("status not set")
        else:
            return self.status_

    @status.setter
    def status(self,value):
        self.status_ = value

    @property
    def baryon_number(self):
        """Get the baryon number of the particle.

        Returns
        -------
        baryon_number_ : int

        Raises
        ------
        ValueError
            if baryon_number is not set
        """
        if self.baryon_number_ == None:
            raise ValueError("baryon number not set")
        else:
            return self.baryon_number_

    @baryon_number.setter
    def baryon_number(self,value):
        self.baryon_number_ = value

    @property
    def strangeness(self):
        """Get the strangeness of the particle.

        Returns
        -------
        strangeness_ : int

        Raises
        ------
        ValueError
            if strangeness is not set
        """
        if self.strangeness_ == None:
            raise ValueError("strangeness not set")
        else:
            return self.strangeness_

    @strangeness.setter
    def strangeness(self,value):
        self.strangeness_ = value

    @property
    def weight(self):
        """Get the weight of the particle.

        Returns
        -------
        weight_ : float

        Raises
        ------
        ValueError
            if weight is not set
        """
        return self.weight_

    @weight.setter
    def weight(self,value):
        self.weight_ = value


    def print_particle(self):
        """Print the whole particle information as csv string.

        This function prints a header line with the different quantities.
        All particle quantities are then printed in the next line separated by
        a comma.
        """
        print('t,x,y,z,mass,E,px,py,pz,pdg,ID,charge,ncoll,form_time,xsecfac,\
              proc_id_origin,proc_type_origin,t_last_coll,pdg_mother1,\
              pdg_mother2,status,baryon_number,weight')
        print(f'{self.t_},{self.x_},{self.y_},{self.z_},{self.mass_},{self.E_},\
              {self.px_},{self.py_},{self.pz_},{self.pdg_},{self.ID_},\
              {self.charge_},{self.ncoll_},{self.form_time_},{self.xsecfac_},\
              {self.proc_id_origin_},{self.proc_type_origin_}\
              ,{self.t_last_coll_},{self.pdg_mother1_},{self.pdg_mother2_},\
              {self.status_},{self.baryon_number_},{self.weight_}')


    def set_quantities_OSCAR2013(self,line_from_file):
        """
        Sets the particle quantities obtained from OSCAR2013.

        Parameters
        ----------
        line_from_file: list, numpy.ndarray
            Contains the values read from the file.

        Raises
        ------
        ValueError
            if the input line has not the same number of columns as OSCAR2013
        """
        # check if the line is a list or numpy array
        if (type(line_from_file) == list or isinstance(line_from_file, np.ndarray))\
              and len(line_from_file)==12:
            self.t = float(line_from_file[0])
            self.x = float(line_from_file[1])
            self.y = float(line_from_file[2])
            self.z = float(line_from_file[3])
            self.mass = float(line_from_file[4])
            self.E = float(line_from_file[5])
            self.px = float(line_from_file[6])
            self.py = float(line_from_file[7])
            self.pz = float(line_from_file[8])
            self.pdg = int(line_from_file[9])
            self.ID = int(line_from_file[10])
            self.charge = int(line_from_file[11])
        else:
            error_message = 'The input line does not have the same number of '+\
                            'columns as the OSCAR2013 format'
            raise ValueError(error_message)

    def set_quantities_OSCAR2013Extended(self,line_from_file,photon=False):
        """
        Sets the particle quantities obtained from OSCAR2013Extended.

        Parameters
        ----------
        line_from_file: list, numpy.ndarray
            Contains the values read from the file.
        photon: bool
            Read in the extra field 'weight' for photon output.

        Raises
        ------
        ValueError
            if the input line has not the same number of columns as OSCAR2013Extended
        """
        # check if the line is a list or numpy array
        if (type(line_from_file) == list or isinstance(line_from_file, np.ndarray))\
              and len(line_from_file)<=22 and len(line_from_file)>=20:
            self.t = float(line_from_file[0])
            self.x = float(line_from_file[1])
            self.y = float(line_from_file[2])
            self.z = float(line_from_file[3])
            self.mass = float(line_from_file[4])
            self.E = float(line_from_file[5])
            self.px = float(line_from_file[6])
            self.py = float(line_from_file[7])
            self.pz = float(line_from_file[8])
            self.pdg = int(line_from_file[9])
            self.ID = int(line_from_file[10])
            self.charge = int(line_from_file[11])
            self.ncoll = int(line_from_file[12])
            self.form_time = float(line_from_file[13])
            self.xsecfac = float(line_from_file[14])
            self.proc_id_origin = int(line_from_file[15])
            self.proc_type_origin = int(line_from_file[16])
            self.t_last_coll = float(line_from_file[17])
            self.pdg_mother1 = int(line_from_file[18])
            self.pdg_mother2 = int(line_from_file[19])
            if len(line_from_file)==22 and not photon:
                self.baryon_number = int(line_from_file[20])
                self.strangeness = int(line_from_file[21])
            elif len(line_from_file)==21 and photon:
                self.weight = float(line_from_file[20])
        else:
            error_message = 'The input line does not have the same number of '+\
                            'columns as the OSCAR2013Extended format'
            raise ValueError(error_message)

    def set_quantities_JETSCAPE(self,line_from_file):
        """
        Sets the particle quantities obtained from a JETSCAPE hadron file.

        The JETSCAPE hadron output does not directly contain information about
        the mass and the charge. Thus, they are computed from the four-momentum
        and the PDG code respectively.

        Parameters
        ----------
        line_from_file: list, numpy.ndarray
            Contains the values read from the file.

        Raises
        ------
        ValueError
            if the input line has not the same number of columns as JETSCAPE hadron file
        """
        # check if the line is a list or numpy array
        if (type(line_from_file) == list or isinstance(line_from_file, np.ndarray))\
              and len(line_from_file)==7:
            self.ID = int(line_from_file[0])
            self.pdg = int(line_from_file[1])
            self.status = int(line_from_file[2])
            self.E = float(line_from_file[3])
            self.px = float(line_from_file[4])
            self.py = float(line_from_file[5])
            self.pz = float(line_from_file[6])

            self.mass = self.compute_mass_from_energy_momentum()
            self.charge = self.compute_charge_from_pdg()
        else:
            error_message = 'The input line does not have the same number of '+\
                            'columns as the JETSCAPE hadron output format'
            raise ValueError(error_message)

    def angular_momentum(self):
        """Compute the angular momentum :math:`L=r \\times p` of a particle

        Returns
        -------
        angular_momentum : numpy.ndarray
            Array containing all three components of the
            angular momentum as :math:`[L_x, L_y, L_z]`.
        """
        r = [self.x, self.y, self.z]
        p = [self.px, self.py, self.pz]

        return np.cross(r, p)

    def momentum_rapidity_Y(self):
        """
        Compute the momentum rapidity :math:`Y=\\frac{1}{2}\\ln\\left(\\frac{E+p_z}{E-p_z}\\right)` of the particle.

        Returns
        -------
        float
            momentum rapidity
        """
        if abs(self.E - self.pz) < 1e-10:
            denominator = (self.E - self.pz) + 1e-10  # Adding a small positive value
        else:
            denominator = (self.E - self.pz)

        return 0.5 * np.log((self.E + self.pz) / denominator)

    def p_abs(self):
        """
        Compute the absolute momentum :math:`|\\vec{p}|=\\sqrt{p_x^2+p_y^2+p_z^2}` of the particle.

        Returns
        -------
        float
            absolute momentum
        """
        return np.sqrt(self.px**2.+self.py**2.+self.pz**2.)

    def pt_abs(self):
        """
        Compute the absolute transverse momentum :math:`|\\vec{p}_{\\mathrm{T}}|=\\sqrt{p_x^2+p_y^2}` of the particle.

        Returns
        -------
        float
            absolute transverse momentum
        """
        return np.sqrt(self.px**2.+self.py**2.)

    def phi(self):
        """
        Compute the azimuthal angle of the particle.

        Returns
        -------
        float
            azimuthal angle
        """
        if (np.abs(self.px) < 1e-6) and (np.abs(self.py) < 1e-6):
            return 0.
        else:
            return math.atan2(self.py,self.px)

    def theta(self):
        """
        Compute the polar angle of the particle.

        Returns
        -------
        float
            polar angle
        """
        if self.p_abs() == 0:
            return 0.
        else:
            return np.arccos(self.pz / self.p_abs())

    def pseudorapidity(self):
        """
        Compute the pseudorapidity :math:`\eta=\\frac{1}{2}\\ln\\left(\\frac{|\\vec{p}|+p_z}{|\\vec{p}|-p_z}\\right)` of the particle.

        Returns
        -------
        float
            pseudorapidity
        """
        if abs(self.p_abs() - self.pz) < 1e-10:
            denominator = (self.p_abs() - self.pz) + 1e-10  # Adding a small positive value
        else:
            denominator = (self.p_abs() - self.pz)

        return 0.5 * np.log((self.p_abs()+self.pz) / denominator)

    def spatial_rapidity(self):
        """
        Compute the spatial rapidity :math:`y=\\frac{1}{2}\\ln\\left(\\frac{t+z}{t-z}\\right)` of the particle.

        Returns
        -------
        float
            spatial rapidity
        """
        if self.t > np.abs(self.z):
            return 0.5 * np.log((self.t+self.z) / (self.t-self.z))
        else:
            raise ValueError("|z| < t not fulfilled")

    def proper_time(self):
        """
        Compute the proper time :math:`\\tau=\\sqrt{t^2-z^2}` of the particle.

        Returns
        -------
        float
            proper time
        """
        if self.t > np.abs(self.z):
            return np.sqrt(self.t**2.-self.z**2.)
        else:
            raise ValueError("|z| < t not fulfilled")

    def compute_mass_from_energy_momentum(self):
        """
        Compute the mass from the energy momentum relation.

        Returns
        -------
        float
            mass
        """
        if np.abs(self.E**2. - self.p_abs()**2.) > 1e-6 and\
              self.E**2. - self.p_abs()**2. > 0.:
            return np.sqrt(self.E**2.-self.p_abs()**2.)
        else:
            return 0.

    def compute_charge_from_pdg(self):
        """
        Compute the charge from the PDG code.

        This function is called automatically if a JETSCAPE file is read in with
        the :meth:`ParticleClass.Particle.set_quantities_JETSCAPE` function.

        Returns
        -------
        float
            charge
        """
        return PDGID(self.pdg).charge

    def is_meson(self):
        """
        Is the particle a meson?

        Returns
        -------
        bool
            True, False
        """
        return PDGID(self.pdg).is_meson

    def is_baryon(self):
        """
        Is the particle a baryon?

        Returns
        -------
        bool
            True, False
        """
        return PDGID(self.pdg).is_baryon

    def is_hadron(self):
        """
        Is the particle a hadron?

        Returns
        -------
        bool
            True, False
        """
        return PDGID(self.pdg).is_hadron

    def is_strange(self):
        """
        Does the particle contain strangeness?

        Returns
        -------
        bool
            True, False
        """
        return PDGID(self.pdg).has_strange

    def is_heavy_flavor(self):
        """
        Is the particle a heavy flavor hadron?

        Returns
        -------
        bool
            True, False
        """
        if PDGID(self.pdg).has_charm or PDGID(self.pdg).has_bottom\
              or PDGID(self.pdg).has_top:
            return True
        else:
            return False
