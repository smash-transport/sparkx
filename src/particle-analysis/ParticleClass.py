import numpy as np
import math
from particle import PDGID

class Particle:
    """
    Defines a particle object.

    The member variables of the Particle class are the quantities in the 
    OSCAR2013/OSCAR2013Extended or JETSCAPE hadron output. If they are not set,
    they stay None to throw an error if one tries to access a non existing 
    quantity.
    """
    def __init__(self):
        self._t = None
        self._x = None
        self._y = None
        self._z = None
        self._mass = None
        self._E = None
        self._px = None
        self._py = None
        self._pz = None
        self._pdg = None
        self._ID = None
        self._charge = None
        self._ncoll = None
        self._form_time = None
        self._xsecfac = None
        self._proc_id_origin = None
        self._proc_type_origin = None
        self._t_last_coll = None
        self._pdg_mother1 = None
        self._pdg_mother2 = None
        self._status = None
        
    # define functions to set the parameters
    def set_t(self,value):
        """Set the time of the particle."""
        self._t = value

    def set_x(self,value):
        """Set the x-position of the particle."""
        self._x = value

    def set_y(self,value):
        """Set the y-position of the particle."""
        self._y = value
    
    def set_z(self,value):
        """Set the z-position of the particle."""
        self._z = value

    def set_mass(self,value):
        """Set the mass of the particle."""
        self._mass = value
    
    def set_E(self,value):
        """
        Set the energy of the particle (zeroth component of 
        momentum four vector).
        """
        self._E = value

    def set_px(self,value):
        """Set the momentum x-component of the particle."""
        self._px = value

    def set_py(self,value):
        """Set the momentum y-component of the particle."""
        self._py = value
    
    def set_pz(self,value):
        """Set the momentum z-component of the particle."""
        self._pz = value

    def set_pdg(self,value):
        """Set the PDG code of the particle."""
        self._pdg = value

    def set_ID(self,value):
        """Set the ID of the particle. This is a unique number in SMASH."""
        self._ID = value

    def set_charge(self,value):
        """Set the electrical charge of the particle."""
        self._charge = value

    def set_ncoll(self,value):
        """Set the number of collisions of the particle."""
        self._ncoll = value

    def set_form_time(self,value):
        """Set the formation time of the particle."""
        self._form_time = value

    def set_xsecfac(self,value):
        """Set the crosssection scaling factor of the particle."""
        self._xsecfac = value

    def set_proc_id_origin(self,value):
        """Set the process ID of the particle's origin."""
        self._proc_id_origin = value

    def set_proc_type_origin(self,value):
        """Set the process type of the particle's origin."""
        self._proc_type_origin = value

    def set_t_last_coll(self,value):
        """Set the last time of a collision of the particle."""
        self._t_last_coll = value

    def set_pdg_mother1(self,value):
        """Set the PDG code of the first mother particle."""
        self._pdg_mother1 = value

    def set_pdg_mother2(self,value):
        """Set the PDG code of the second mother particle."""
        self._pdg_mother2 = value

    def set_status(self,value):
        """
        Set the hadron status (stores information on the module origin of 
        a JETSCAPE hadron).
        """
        self._status = value

    # define functions to get the parameters
    def t(self):
        """Get the time of the particle."""
        if self._t == None:
            raise ValueError("t not set")
        else:
            return self._t

    def x(self):
        """Get the x-position of the particle."""
        if self._x == None:
            raise ValueError("x not set")
        else:
            return self._x

    def y(self):
        """Get the y-position of the particle."""
        if self._y == None:
            raise ValueError("y not set")
        else:
            return self._y
    
    def z(self):
        """Get the z-position of the particle."""
        if self._z == None:
            raise ValueError("z not set")
        else:
            return self._z

    def mass(self):
        """Get the mass of the particle."""
        if self._mass == None:
            raise ValueError("mass not set")
        else:
            return self._mass
    
    def E(self):
        """
        Get the energy of the particle (zeroth component of 
        momentum four vector).
        """
        if self._E == None:
            raise ValueError("E not set")
        else:
            return self._E

    def px(self):
        """Get the momentum x-component of the particle."""
        if self._px == None:
            raise ValueError("px not set")
        else:
            return self._px

    def py(self):
        """Get the momentum y-component of the particle."""
        if self._py == None:
            raise ValueError("py not set")
        else:
            return self._py
    
    def pz(self):
        """Get the momentum z-component of the particle."""
        if self._pz == None:
            raise ValueError("pz not set")
        else:
            return self._pz

    def pdg(self):
        """Get the PDG code of the particle."""
        if self._pdg == None:
            raise ValueError("pdg not set")
        else:
            return self._pdg

    def ID(self):
        """Get the ID of the particle. This is a unique number in SMASH."""
        if self._ID == None:
            raise ValueError("ID not set")
        else:
            return self._ID

    def charge(self):
        """Get the electrical charge of the particle."""
        if self._charge == None:
            raise ValueError("charge not set")
        else:
            return self._charge

    def ncoll(self):
        """Get the number of collisions of the particle."""
        if self._ncoll == None:
            raise ValueError("ncoll not set")
        else:
            return self._ncoll

    def form_time(self):
        """Get the formation time of the particle."""
        if self._form_time == None:
            raise ValueError("form_time not set")
        else:
            return self._form_time

    def xsecfac(self):
        """Get the crosssection scaling factor of the particle."""
        if self._xsecfac == None:
            raise ValueError("xsecfac not set")
        else:
            return self._xsecfac

    def proc_id_origin(self):
        """Get the process ID of the particle's origin."""
        if self._proc_id_origin == None:
            raise ValueError("proc_id_origin not set")
        else:
            return self._proc_id_origin

    def proc_type_origin(self):
        """Get the process type of the particle's origin."""
        if self._proc_type_origin == None:
            raise ValueError("proc_type_origin not set")
        else:
            return self._proc_type_origin

    def t_last_coll(self):
        """Get the last time of a collision of the particle."""
        if self._t_last_coll == None:
            raise ValueError("t_last_coll not set")
        else:
            return self._t_last_coll

    def pdg_mother1(self):
        """Get the PDG code of the first mother particle."""
        if self._pdg_mother1 == None:
            raise ValueError("pdg_mother1 not set")
        else:
            return self._pdg_mother1

    def pdg_mother2(self):
        """Get the PDG code of the second mother particle."""
        if self._pdg_mother2 == None:
            raise ValueError("pdg_mother2 not set")
        else:
            return self._pdg_mother2
        
    def status(self):
        """
        Get the hadron status (stores information on the module origin of 
        a JETSCAPE hadron).
        """
        if self._status == None:
            raise ValueError("status not set")
        else:
            return self._status
    
    def print_particle(self):
        """Print the whole particle information as csv string."""
        print('t,x,y,z,mass,E,px,py,pz,pdg,ID,charge,ncoll,form_time,xsecfac,\
              proc_id_origin,proc_type_origin,t_last_coll,pdg_mother1,\
              pdg_mother2,status')
        print(f'{self._t},{self._x},{self._y},{self._z},{self._mass},{self._E},\
              {self._px},{self._py},{self._pz},{self._pdg},{self._ID},\
              {self._charge},{self._ncoll},{self._form_time},{self._xsecfac},\
              {self._proc_id_origin},{self._proc_type_origin},{self._t_last_coll},\
              {self._pdg_mother1},{self._pdg_mother2},{self._status}')


    def set_quantities_OSCAR2013(self,line_from_file):
        """
        Sets the particle quantities obtained from OSCAR2013.

        Parameters
        ----------
        line_from_file: list, numpy.ndarray
            Contains the values read from the file.
        """
        # check if the line is a list or numpy array
        if (type(line_from_file) == list or isinstance(line_from_file, np.ndarray))\
              and len(line_from_file)==12:
            self.set_t(line_from_file[0])
            self.set_x(line_from_file[1])
            self.set_y(line_from_file[2])
            self.set_z(line_from_file[3])
            self.set_mass(line_from_file[4])
            self.set_E(line_from_file[5])
            self.set_px(line_from_file[6])
            self.set_py(line_from_file[7])
            self.set_pz(line_from_file[8])
            self.set_pdg(line_from_file[9])
            self.set_ID(line_from_file[10])
            self.set_charge(line_from_file[11])
        else:
            error_message = 'The input line does not have the same number of '+\
                            'columns as the OSCAR2013 format'
            raise ValueError(error_message)

    def set_quantities_OSCAR2013Extended(self,line_from_file):
        """
        Sets the particle quantities obtained from OSCAR2013Extended.

        Parameters
        ----------
        line_from_file: list, numpy.ndarray
            Contains the values read from the file.
        """
        # check if the line is a list or numpy array
        if (type(line_from_file) == list or isinstance(line_from_file, np.ndarray))\
              and len(line_from_file)==20:
            self.set_t(line_from_file[0])
            self.set_x(line_from_file[1])
            self.set_y(line_from_file[2])
            self.set_z(line_from_file[3])
            self.set_mass(line_from_file[4])
            self.set_E(line_from_file[5])
            self.set_px(line_from_file[6])
            self.set_py(line_from_file[7])
            self.set_pz(line_from_file[8])
            self.set_pdg(line_from_file[9])
            self.set_ID(line_from_file[10])
            self.set_charge(line_from_file[11])
            self.set_ncoll(line_from_file[12])
            self.set_form_time(line_from_file[13])
            self.set_xsecfac(line_from_file[14])
            self.set_proc_id_origin(line_from_file[15])
            self.set_proc_type_origin(line_from_file[16])
            self.set_t_last_coll(line_from_file[17])
            self.set_pdg_mother1(line_from_file[18])
            self.set_pdg_mother2(line_from_file[19])
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
        """
        # check if the line is a list or numpy array
        if (type(line_from_file) == list or isinstance(line_from_file, np.ndarray))\
              and len(line_from_file)==7:
            self.set_ID(line_from_file[0])
            self.set_pdg(line_from_file[1])
            self.set_status(line_from_file[2])
            self.set_E(line_from_file[3])
            self.set_px(line_from_file[4])
            self.set_py(line_from_file[5])
            self.set_pz(line_from_file[6])

            self.set_mass(self.compute_mass_from_energy_momentum())
            self.set_charge(self.compute_charge_from_pdg())
        else:
            error_message = 'The input line does not have the same number of '+\
                            'columns as the JETSCAPE hadron output format'
            raise ValueError(error_message)
        
    def momentum_rapidity_Y(self):
        """
        Compute the momentum rapidity Y of the particle.
        
        Returns
        -------
        float
            momentum rapidity
        """
        return 0.5 * np.log((self.E + self.pz()) / (self.E - self.pz()))
    
    def p_abs(self):
        """
        Compute the absolute momentum of the particle.
        
        Returns
        -------
        float
            absolute momentum
        """
        return np.sqrt(self.px()**2.+self.py()**2.+self.pz()**2.)

    def pt_abs(self):
        """
        Compute the transverse momentum of the particle.
        
        Returns
        -------
        float
            absolute transverse momentum
        """
        return np.sqrt(self.px()**2.+self.py()**2.)
    
    def phi(self):
        """
        Compute the azimuthal angle of the particle.
        
        Returns
        -------
        float
            azimuthal angle
        """
        if (np.abs(self.px()) < 1e-6) and (np.abs(self.px()) < 1e-6):
            return 0.
        else:
            return math.atan2(self.py(),self.px())

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
            return np.arccos(self.pz() / self.p_abs())

    def pseudorapidity(self):
        """
        Compute the pseudorapidity of the particle.
        
        Returns
        -------
        float
            pseudorapidity
        """
        return 0.5 * np.log((self.p_abs()+self.pz()) / (self.p_abs()-self.pz()))
    
    def spatial_rapidity(self):
        """
        Compute the spatial rapidity of the particle.
        
        Returns
        -------
        float
            spatial rapidity
        """
        if self.t() > np.abs(self.z()):
            return 0.5 * np.log((self.t()+self.z()) / (self.t()-self.z()))
        else:
            raise ValueError("|z| < t not fulfilled")
        
    def proper_time(self):
        """
        Compute the proper time of the particle.
        
        Returns
        -------
        float
            proper time
        """
        if self.t() > np.abs(self.z()):
            return np.sqrt(self.t()**2.-self.z()**2.)
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
        if self.E()**2. > self.p_abs()**2.:
            return np.sqrt(self.E()**2.-self.p_abs()**2.)
        else:
            raise ValueError("Can not compute mass from on-shell condition")
        
    def compute_charge_from_pdg(self):
        """
        Compute the charge from the PDG code.

        Returns
        -------
        float 
            charge
        """
        return PDGID(self.pdg()).charge
    
    def is_meson(self):
        """
        Is the particle a meson?

        Returns
        -------
        bool
            True, False
        """
        return PDGID(self.pdg()).is_meson
    
    def is_baryon(self):
        """
        Is the particle a baryon?

        Returns
        -------
        bool
            True, False
        """
        return PDGID(self.pdg()).is_baryon
    
    def is_hadron(self):
        """
        Is the particle a hadron?

        Returns
        -------
        bool
            True, False
        """
        return PDGID(self.pdg()).is_hadron
    
    def is_strange(self):
        """
        Does the particle contain strangeness?

        Returns
        -------
        bool
            True, False
        """
        return PDGID(self.pdg()).has_strange
    
    def is_heavy_flavor(self):
        """
        Is the particle a heavy flavor hadron?

        Returns
        -------
        bool
            True, False
        """
        if PDGID(self.pdg()).has_charm or PDGID(self.pdg()).has_bottom\
              or PDGID(self.pdg()).has_top:
            return True
        else:
            return False