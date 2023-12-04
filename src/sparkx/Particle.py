import numpy as np
import math
from particle import PDGID
import warnings

class Particle:
    """Defines a particle object.

    The member variables of the Particle class are the quantities in the
    OSCAR2013/OSCAR2013Extended or JETSCAPE hadron output. If they are not set,
    they stay None to throw an error if one tries to access a non existing
    quantity.
    If a particle with an unknown PDG is provided, a warning is thrown and and 
    None is returned for charge, spin, and spin degeneracy.

    Attributes
    ----------
    t : float
        The time of the particle.
    x : float
        The x coordinate of the position.
    y : float
        The y coordinate of the position.
    z : float
        The z coordinate of the position.
    mass : float
        The mass of the particle.
    E : float
        The energy of the particle.
    px : float
        The x component of the momentum.
    py : float
        The y component of the momentum.
    pz : float
        The z component of the momentum.
    pdg : int
        The PDG code of the particle.
    pdg_is_valid : short
        Is the PDG code valid?
    ID : int
        The ID of the particle (unique label of each particle in an event).
    charge : short
        Electric charge of the particle.
    ncoll : int
        Number of collisions undergone by the particle.
    form_time : double
        Formation time of the particle.
    xsecfac : double
        Scaling factor of the cross section.
    proc_id_origin : int
        ID for the process the particle stems from.
    proc_type_origin : int
        Type information for the process the particle stems from.
    t_last_coll : double
        Time of the last collision.
    pdg_mother1 : int
        PDG code of the mother particle 1.
    pdg_mother2 : int
        PDG code of the mother particle 2.
    status : int
        Status code of the particle.
    baryon_number : short
        Baryon number of the particle.
    strangeness : short
        Strangeness quantum number of the particle.
    weight : float
        Weight of the particle.

    These attributes are saved in following data structure:

    Data
    -----

    data_float_ :
        The float data values in following order:
        t, x, y, z, mass, E, px, py, pz, form_time, xsecfac, t_last_coll, weight
    data_int_ :
        The int data values in following order:
        pdg, ID, ncoll, proc_id_orgin, pdg_mother1, pdg_mother2, status 
    data_short_ :
        The short data values in following order:
        pdg_valid, charge, baryon_number, strangeness



    Methods
    -------
    t :
        Get/set t
    x :
        Get/set x
    y :
        Get/set y
    z :
        Get/set z
    mass :
        Get/set mass
    E :
        Get/set E
    px :
        Get/set px
    py :
        Get/set py
    pz :
        Get/set pz
    pdg :
        Get/set pdg
    ID :
        Get/set ID
    charge :
        Get/set charge
    ncoll :
        Get/set ncoll
    form_time :
        Get/set form_time
    xsecfac :
        Get/set xsecfactor
    proc_id_origin :
        Get/set proc_id_origin
    proc_type_origin :
        Get/set proc_type_origin
    t_last_coll :
        Get/set t_last_coll
    pdg_mother1 :
        Get/set pdg_mother1
    pdg_mother2 :
        Get/set pdg_mother2
    status :
        Get/set status
    baryon_number :
        Get/set baryon_number
    strangeness :
        Get/set strangeness
    print_particle :
        Print the particle as CSV to terminal
    angular_momentum :
        Compute angular momentum
    momentum_rapidity_Y :
        Compute momentum rapidity
    p_abs :
        Compute absolute momentum
    pt_abs :
        Compute absolute value of transverse momentum
    phi :
        Compute azimuthal angle
    theta :
        Compute polar angle
    pseudorapidity :
        Compute pseudorapidity
    spatial_rapidity :
        Compute spatial rapidity
    proper_time :
        Compute proper time
    compute_mass_from_energy_momentum :
        Compute mass from energy momentum relation
    compute_charge_from_pdg :
        Compute charge from PDG code
    is_meson :
        Is the particle a meson?
    is_baryon :
        Is the particle a baryon?
    is_hadron :
        Is the particle a hadron?
    is_strange :
        Is the particle a strange particle?
    is_heavy_flavor :
        Is the particle a heavy flavor particle?
    weight :
        What is the weight of the particle?
    spin :
        Total spin :math:`J` of the particle.
    spin_degeneracy :
        Total spin :math:`2J + 1` of the particle.
    

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

    The class can be used to construct a particle from different input formats.
    Supported formats include:
        - "Oscar2013"
        - "Oscar2013Extended"
        - "Oscar2013Extended_IC"
        - "Oscar2013Extended_Photons"
        - "JETSCAPE"
    
    .. highlight:: python
    .. code-block:: python
        :linenos:

        >>> particle_quantity_JETSCAPE = np.array([0,2114,11,2.01351754,1.30688601,-0.422958786,-0.512249773])
        >>> particle = Particle(input_format="JETSCAPE", particle_array=particle_array_oscar2013)

    Notes
    -----
    If a member of the Particle class is not set or a quantity should be computed
    and the needed member variables are not set, then `None` is returned by default.
    """
    __slots__ = ['data_float_','data_int_', 'data_short_'] 
    def __init__(self,input_format=None,particle_array=None):
        # t, x, y, z, mass, E, px, py, pz, form_time_, xsecfac_, t_last_coll_, weight_
        self.data_float_ = np.array(13*[None],dtype = float)
        # pdg, ID, ncoll, proc_id_orgin, pdg_mother1, pdg_mother2, status 
        self.data_int_ = np.array(7*[None], dtype = int)
        # pdg_valid, charge, baryon_number, strangeness
        self.data_short_ = np.array(4*[None], dtype = short) 

        if ((input_format is not None) and (particle_array is None)) or ((input_format is None) and (particle_array is not None)):
            raise ValueError("'input_format' or 'particle_array' not given")

        if (input_format is not None) and (particle_array is not None):
            self.__initialize_from_array(input_format,particle_array)
            
        
    def __initialize_from_array(self,input_format,particle_array):
        """
        Initialize instance attributes based on the provided input format and array.

        Parameters
        ----------
        input_format : str
            The format of the input data. Supported formats include:
            - "Oscar2013"
            - "Oscar2013Extended"
            - "Oscar2013Extended_IC"
            - "Oscar2013Extended_Photons"
            - "JETSCAPE"

        particle_array : numpy.ndarray
            An array containing particle information.

        Raises
        ------
        ValueError
            If the input format is unsupported or the array length is invalid.

        Notes
        -----
        For each supported input format, this method expects a specific order 
        of elements in the particle_array and assigns them to the corresponding 
        attributes of the Particle instance.

        If the input format is "JETSCAPE," additional attributes (mass_ and 
        charge_) are computed based on energy-momentum and PDG code.

        """
        #first entry: index in data array
        #second entry: index in line
        attribute_mapping = {
            "Oscar2013": {
                "t_": [0,0],
                "x_": [1,1],
                "y_": [2,2],
                "z_": [3,3],
                "mass_": [4,4],
                "E_": [5,5],
                "px_": [6,6],
                "py_": [7,7],
                "pz_": [8,8],
                "pdg_": [0,9],
                "ID_": [1,10],
                "charge_": [1,11],
            },
            "Oscar2013Extended": {
                "t_": [0,0],
                "x_": [1,1],
                "y_": [2,2],
                "z_": [3,3],
                "mass_": [4,4],
                "E_": [5,5],
                "px_": [6,6],
                "py_": [7,7],
                "pz_": [8,8],
                "pdg_": [0,9],
                "ID_": [1,10],
                "charge_": [1,11],
                "ncoll_": [2,12],
                "form_time_": [9,11],
                "xsecfac_": [10,14],
                "proc_id_origin_": [3,15],
                "proc_type_origin_": [4,16],
                "t_last_coll_": [11,17],
                "pdg_mother1_": [4,18],
                "pdg_mother2_": [5,19],
                "baryon_number_": [2,20],
                "strangeness_": [3,21],
            },
            "Oscar2013Extended_IC": {
                "t_": [0,0],
                "x_": [1,1],
                "y_": [2,2],
                "z_": [3,3],
                "mass_": [4,4],
                "E_": [5,5],
                "px_": [6,6],
                "py_": [7,7],
                "pz_": [8,8],
                "pdg_": [0,9],
                "ID_": [1,10],
                "charge_": [1,11],
                "ncoll_": [2,12],
                "form_time_": [9,11],
                "xsecfac_": [10,14],
                "proc_id_origin_": [3,15],
                "proc_type_origin_": [4,16],
                "t_last_coll_": [11,17],
                "pdg_mother1_": [4,18],
                "pdg_mother2_": [5,19],
                "baryon_number_": [2,20],
                "strangeness_": [3,21],
            },
            "Oscar2013Extended_Photons": {
                "t_": [0,0],
                "x_": [1,1],
                "y_": [2,2],
                "z_": [3,3],
                "mass_": [4,4],
                "E_": [5,5],
                "px_": [6,6],
                "py_": [7,7],
                "pz_": [8,8],
                "pdg_": [0,9],
                "ID_": [1,10],
                "charge_": [1,11],
                "ncoll_": [2,12],
                "form_time_": [9,11],
                "xsecfac_": [10,14],
                "proc_id_origin_": [3,15],
                "proc_type_origin_": [4,16],
                "t_last_coll_": [11,17],
                "pdg_mother1_": [4,18],
                "pdg_mother2_": [5,19],
                "weight_": [12,20],
            },
            "JETSCAPE": {
                "ID_": [1,10],
                "pdg_": [0,9],
                "status_": [6,2],
                "E_": [5,3],
                "px_": [6,4],
                "py_": [7,5],
                "pz_": [8,6],
            },
        }
        if (input_format in attribute_mapping):
            if (len(particle_array) == len(attribute_mapping[input_format]) or (input_format in ["Oscar2013Extended","Oscar2013Extended_IC"]\
                 and len(particle_array) <=  len(attribute_mapping[input_format])\
                    and len(particle_array) >=  len(attribute_mapping[input_format])-2)):
                for attribute, index in attribute_mapping[input_format].items():
                    if len(particle_array)<index[1]+1:
                        setattr(self,attribute,None)
                        continue
                    # Type casting for specific attributes
                    if attribute in ["t_", "x_", "y_", "z_", "mass_", "E_", "px_", "py_", "pz_", "form_time_", "xsecfac_", "t_last_coll_", "weight_"]:
                        self.data_float_[index[0]] = float(particle_array[index[1]])
                    elif attribute in ["pdg_", "ID_", "ncoll_", "proc_id_origin_", "proc_type_origin_", "pdg_mother1_", "pdg_mother2_", "status_"]:
                        self.data_int_[index[0]] = int(particle_array[index[1]])
                    else:
                        self.data_short_[index[0]] = int(particle_array[index[1]])

                if input_format == "JETSCAPE":
                    self.mass_ = self.compute_mass_from_energy_momentum()
                    self.charge_ = self.compute_charge_from_pdg()
            else:
                raise ValueError(f"The input file is corrupted!")
        else:
            raise ValueError(f"Unsupported input format '{input_format}'")
        
        self.pdg_valid_ = PDGID(self.pdg_).is_valid

        if(not self.pdg_valid_):
             warnings.warn('The PDG code ' + str(self.pdg_) + ' is not valid. '+
                           'All properties extracted from the PDG are set to default values.')

    @property
    def t(self):
        """Get or set the time of the particle.

        Returns
        -------
        t_ : float
        """
        return self.data_float_[0]

    @t.setter
    def t(self,value):
        self.data_float_[0] = value

    @property
    def x(self):
        """Get or set the x-position of the particle.

        Returns
        -------
        x_ : float
        """
        return self.data_float_[1]

    @x.setter
    def x(self,value):
        self.data_float_[1] = value

    @property
    def y(self):
        """Get or set the y-position of the particle.

        Returns
        -------
        y_ : float
        """
        return self.data_float_[2]

    @y.setter
    def y(self,value):
        self.data_float_[2] = value

    @property
    def z(self):
        """Get or set the z-position of the particle.

        Returns
        -------
        z_ : float
        """
        return self.data_float_[3]
    @z.setter
    def z(self,value):
        self.data_float_[4] = value

    @property
    def mass(self):
        """Get or set the mass of the particle.

        Returns
        -------
        mass_ : float
        """
        return self.data_float_[5]

    @mass.setter
    def mass(self,value):
        self.data_float_[5] = value

    @property
    def E(self):
        """Get or set the energy of the particle.

        Returns
        -------
        E_ : float
        """
        return self.data_float_[6]

    @E.setter
    def E(self,value):
        self.data_float_[6] = value

    @property
    def px(self):
        """Get or set the momentum x-component of the particle.

        Returns
        -------
        px_ : float
        """
        return self.data_float_[7]

    @px.setter
    def px(self,value):
        self.data_float_[7] = value

    @property
    def py(self):
        """Get or set the momentum y-component of the particle.

        Returns
        -------
        py_ : float
        """
        return self.data_float_[8]

    @py.setter
    def py(self,value):
        self.data_float_[8] = value

    @property
    def pz(self):
        """Get or set the momentum z-component of the particle.

        Returns
        -------
        pz_ : float
        """
        return self.data_float_[9]

    @pz.setter
    def pz(self,value):
        self.data_float_[9] = value

    @property
    def pdg(self):
        """Get or set the PDG code of the particle.

        Returns
        -------
        pdg_ : int
        """
        return self.data_int_[0]

    @pdg.setter
    def pdg(self,value):
        self.data_int_[0] = value
        self.data_short_[0] = PDGID(self.pdg).is_valid

        if(not self.pdg_valid):
             warnings.warn('The PDG code ' + str(self.pdg) + ' is not valid. '+
                           'All properties extracted from the PDG are set to None.')

    @property
    def ID(self):
        """Get or set the ID of the particle.

        This is a unique number in SMASH.

        Returns
        -------
        ID_ : int
        """
        return self.data_int_[1]

    @ID.setter
    def ID(self,value):
        self.data_int_[1] = value

    @property
    def charge(self):
        """Get or set the electrical charge of the particle.

        Returns
        -------
        charge_ : short
        """
        return self.data_short_[1]

    @charge.setter
    def charge(self,value):
        self.data_short_[1] = value

    @property
    def ncoll(self):
        """Get or set the number of collisions of the particle.

        Returns
        -------
        ncoll_ : int
        """
        return self.data_int_[2] 

    @ncoll.setter
    def ncoll(self,value):
        self.data_int_[2]  = value

    @property
    def form_time(self):
        """Get or set the formation time of the particle.

        Returns
        -------
        form_time_ : float
        """
        return self.data_float_[9] 

    @form_time.setter
    def form_time(self,value):
        self.data_float_[9]  = value

    @property
    def xsecfac(self):
        """Get or set the crosssection scaling factor of the particle.

        Returns
        -------
        xsecfac_ : float
        """
        return self.data_float_[10] 

    @xsecfac.setter
    def xsecfac(self,value):
        self.data_float_[10]  = value

    @property
    def proc_id_origin(self):
        """Get or set the process ID of the particle's origin.

        Returns
        -------
        proc_id_origin_ : int
        """
        return self.data_int_[3]

    @proc_id_origin.setter
    def proc_id_origin(self,value):
        self.data_int_[3] = value

    @property
    def proc_type_origin(self):
        """Get or set the process type of the particle's origin.

        Returns
        -------
        proc_type_origin_ : int
        """
        return self.data_int_[4]

    @proc_type_origin.setter
    def proc_type_origin(self,value):
        self.data_int_[4] = value

    @property
    def t_last_coll(self):
        """Get or set the last time of a collision of the particle.

        Returns
        -------
        t_last_coll_ : float
        """
        return self.data_float_[11]

    @t_last_coll.setter
    def t_last_coll(self,value):
        self.data_float_[11] = value

    @property
    def pdg_mother1(self):
        """Get the PDG code of the first mother particle.

        Returns
        -------
        pdg_mother1_ : int
        """
        return self.data_int_[5]

    @pdg_mother1.setter
    def pdg_mother1(self,value):
        self.data_int_[5] = value

    @property
    def pdg_mother2(self):
        """Get the PDG code of the second mother particle.

        Returns
        -------
        pdg_mother2_ : int
        """
        return self.data_int_[6]

    @pdg_mother2.setter
    def pdg_mother2(self,value):
        self.data_int_[6] = value

    @property
    def status(self):
        """
        Get the hadron status (stores information on the module origin of
        a JETSCAPE hadron).

        Returns
        -------
        status_ : int
        """
        return self.data_int_[7]

    @status.setter
    def status(self,value):
        self.data_int_[7] = value

    @property
    def baryon_number(self):
        """Get the baryon number of the particle.

        Returns
        -------
        baryon_number_ : short
        """
        return self.data_short_[2]
 
    @baryon_number.setter
    def baryon_number(self,value):
        self.data_short_[2] = value

    @property
    def strangeness(self):
        """Get the strangeness of the particle.

        Returns
        -------
        strangeness_ : short
        """
        return self.data_short_[3]

    @strangeness.setter
    def strangeness(self,value):
        self.data_short_[3] = value

    @property
    def weight(self):
        """Get the weight of the particle.

        Returns
        -------
        weight_ : float
        """
        return self.data_float_[12]

    @weight.setter
    def weight(self,value):
        self.data_float_[12] = value

    def print_particle(self):
        """Print the whole particle information as csv string.

        This function prints a header line with the different quantities.
        All particle quantities are then printed in the next line separated by
        a comma.
        """
        print('t,x,y,z,mass,E,px,py,pz,pdg,ID,charge,ncoll,form_time,xsecfac,\
              proc_id_origin,proc_type_origin,t_last_coll,pdg_mother1,\
              pdg_mother2,status,baryon_number,strangeness,weight')
        print(f'{self.t},{self.x},{self.y},{self.z},{self.mass},{self.E},\
              {self.px},{self.py},{self.pz},{self.pdg},{self.ID},\
              {self.charge},{self.ncoll},{self.form_time},{self.xsecfac},\
              {self.proc_id_origin},{self.proc_type_origin}\
              ,{self.t_last_coll},{self.pdg_mother1},{self.pdg_mother2},\
              {self.status},{self.baryon_number},{self.strangeness},{self.weight}')

    def angular_momentum(self):
        """
        Compute the angular momentum :math:`L=r \\times p` of a particle.

        Returns
        -------
        angular_momentum : numpy.ndarray
            Array containing all three components of the
            angular momentum as :math:`[L_x, L_y, L_z]`.

        Notes
        -----
        If one of the needed particle quantities is not given, then `None`
        is returned. 
        """
        if (self.x == None) or (self.y == None) or (self.z == None)\
            (self.px == None) or (self.py == None) or (self.pz == None):
            return None
        else:
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
        
        Notes
        -----
        If one of the needed particle quantities is not given, then `None`
        is returned.
        """
        if (self.E == None) or (self.pz == None):
            return None
        else:
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
        
        Notes
        -----
        If one of the needed particle quantities is not given, then `None`
        is returned.
        """
        if (self.px == None) or (self.py == None) or (self.pz == None):
            return None
        else:
            return np.sqrt(self.px**2.+self.py**2.+self.pz**2.)

    def pt_abs(self):
        """
        Compute the absolute transverse momentum :math:`|\\vec{p}_{\\mathrm{T}}|=\\sqrt{p_x^2+p_y^2}` of the particle.

        Returns
        -------
        float
            absolute transverse momentum
        
        Notes
        -----
        If one of the needed particle quantities is not given, then `None`
        is returned.
        """
        if (self.px == None) or (self.py == None):
            return None
        else:
            return np.sqrt(self.px**2.+self.py**2.)

    def phi(self):
        """
        Compute the azimuthal angle of the particle.

        Returns
        -------
        float
            azimuthal angle

        Notes
        -----
        If one of the needed particle quantities is not given, then `None`
        is returned.
        """
        if (self.px == None) or (self.py == None):
            return None
        else:
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

        Notes
        -----
        If one of the needed particle quantities is not given, then `None`
        is returned.
        """
        if (self.px == None) or (self.py == None) or (self.pz == None):
            return None
        else:
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

        Notes
        -----
        If one of the needed particle quantities is not given, then `None`
        is returned.
        """
        if (self.px == None) or (self.py == None) or (self.pz == None):
            return None
        else:
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

        Notes
        -----
        If one of the needed particle quantities is not given, then `None`
        is returned.
        """
        if (self.t == None) or (self.z == None):
            return None
        else:
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

        Notes
        -----
        If one of the needed particle quantities is not given, then `None`
        is returned.
        """
        if (self.t == None) or (self.z == None):
            return None
        else:
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

        Notes
        -----
        If one of the needed particle quantities is not given, then `None`
        is returned.
        """
        if (self.E == None) or (self.px == None) or (self.py == None) or (self.pz == None):
            return None
        else:
            if np.abs(self.E**2. - self.p_abs()**2.) > 1e-16 and\
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
        
        Notes
        -----
        If the PDG ID is not known by `PDGID`, then `None` is returned.
        """
        if not self.pdg_valid_:
            return None
        return PDGID(self.pdg).charge

    def is_meson(self):
        """
        Is the particle a meson?

        Returns
        -------
        bool
            True, False

        Notes
        -----
        If the PDG ID is not known by `PDGID`, then `None` is returned.
        """
        if not self.pdg_valid_:
            return None
        return PDGID(self.pdg).is_meson

    def is_baryon(self):
        """
        Is the particle a baryon?

        Returns
        -------
        bool
            True, False

        Notes
        -----
        If the PDG ID is not known by `PDGID`, then `None` is returned.
        """
        if not self.pdg_valid_:
            return None
        return PDGID(self.pdg).is_baryon

    def is_hadron(self):
        """
        Is the particle a hadron?

        Returns
        -------
        bool
            True, False

        Notes
        -----
        If the PDG ID is not known by `PDGID`, then `None` is returned.
        """
        if not self.pdg_valid_:
            return None
        return PDGID(self.pdg).is_hadron

    def is_strange(self):
        """
        Does the particle contain strangeness?

        Returns
        -------
        bool
            True, False
        
        Notes
        -----
        If the PDG ID is not known by `PDGID`, then `None` is returned.
        """
        if not self.pdg_valid_:
            return None
        return PDGID(self.pdg).has_strange

    def is_heavy_flavor(self):
        """
        Is the particle a heavy flavor hadron?

        Returns
        -------
        bool
            True, False

        Notes
        -----
        If the PDG ID is not known by `PDGID`, then `None` is returned.
        """
        if not self.pdg_valid_:
            return None
        if PDGID(self.pdg).has_charm or PDGID(self.pdg).has_bottom\
              or PDGID(self.pdg).has_top:
            return True
        else:
            return False
     
    def spin(self):
        """
        Get the total spin :math:`J` of the particle.

        Returns
        -------
        float
            Total spin :math:`J`
        
        Notes
        -----
        If the PDG ID is not known by `PDGID`, then `None` is returned.
        """
        if not self.pdg_valid_:
            return None
        return PDGID(self.pdg).J
           
    def spin_degeneracy(self):
        """
        Get the number of all possible spin projections (:math:`2J + 1`).

        Returns
        -------
        int
            Spin degeneracy :math:`2J + 1`

        Notes
        -----
        If the PDG ID is not known by `PDGID`, then `None` is returned.
        """
        if not self.pdg_valid_:
            return None
        return PDGID(self.pdg).j_spin
    