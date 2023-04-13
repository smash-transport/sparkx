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
        
    # define functions to set the parameters
    def set_t(self,value):
        """Set the time of the particle."""
        self.t_ = value

    def set_x(self,value):
        """Set the x-position of the particle."""
        self.x_ = value

    def set_y(self,value):
        """Set the y-position of the particle."""
        self.y_ = value
    
    def set_z(self,value):
        """Set the z-position of the particle."""
        self.z_ = value

    def set_mass(self,value):
        """Set the mass of the particle."""
        self.mass_ = value
    
    def set_E(self,value):
        """
        Set the energy of the particle (zeroth component of 
        momentum four vector).
        """
        self.E_ = value

    def set_px(self,value):
        """Set the momentum x-component of the particle."""
        self.px_ = value

    def set_py(self,value):
        """Set the momentum y-component of the particle."""
        self.py_ = value
    
    def set_pz(self,value):
        """Set the momentum z-component of the particle."""
        self.pz_ = value

    def set_pdg(self,value):
        """Set the PDG code of the particle."""
        self.pdg_ = value

    def set_ID(self,value):
        """Set the ID of the particle. This is a unique number in SMASH."""
        self.ID_ = value

    def set_charge(self,value):
        """Set the electrical charge of the particle."""
        self.charge_ = value

    def set_ncoll(self,value):
        """Set the number of collisions of the particle."""
        self.ncoll_ = value

    def set_form_time(self,value):
        """Set the formation time of the particle."""
        self.form_time_ = value

    def set_xsecfac(self,value):
        """Set the crosssection scaling factor of the particle."""
        self.xsecfac_ = value

    def set_proc_id_origin(self,value):
        """Set the process ID of the particle's origin."""
        self.proc_id_origin_ = value

    def set_proc_type_origin(self,value):
        """Set the process type of the particle's origin."""
        self.proc_type_origin_ = value

    def set_t_last_coll(self,value):
        """Set the last time of a collision of the particle."""
        self.t_last_coll_ = value

    def set_pdg_mother1(self,value):
        """Set the PDG code of the first mother particle."""
        self.pdg_mother1_ = value

    def set_pdg_mother2(self,value):
        """Set the PDG code of the second mother particle."""
        self.pdg_mother2_ = value

    def set_status(self,value):
        """
        Set the hadron status (stores information on the module origin of 
        a JETSCAPE hadron).
        """
        self.status_ = value
        
    def set_baryon_number(self,value):
        """Set the baryon number of the particle."""
        self.baryon_number_ = value

    # define functions to get the parameters
    def t(self):
        """Get the time of the particle."""
        if self.t_ == None:
            raise ValueError("t not set")
        else:
            return self.t_

    def x(self):
        """Get the x-position of the particle."""
        if self.x_ == None:
            raise ValueError("x not set")
        else:
            return self.x_

    def y(self):
        """Get the y-position of the particle."""
        if self.y_ == None:
            raise ValueError("y not set")
        else:
            return self.y_
    
    def z(self):
        """Get the z-position of the particle."""
        if self.z_ == None:
            raise ValueError("z not set")
        else:
            return self.z_

    def mass(self):
        """Get the mass of the particle."""
        if self.mass_ == None:
            raise ValueError("mass not set")
        else:
            return self.mass_
    
    def E(self):
        """
        Get the energy of the particle (zeroth component of 
        momentum four vector).
        """
        if self.E_ == None:
            raise ValueError("E not set")
        else:
            return self.E_

    def px(self):
        """Get the momentum x-component of the particle."""
        if self.px_ == None:
            raise ValueError("px not set")
        else:
            return self.px_

    def py(self):
        """Get the momentum y-component of the particle."""
        if self.py_ == None:
            raise ValueError("py not set")
        else:
            return self.py_
    
    def pz(self):
        """Get the momentum z-component of the particle."""
        if self.pz_ == None:
            raise ValueError("pz not set")
        else:
            return self.pz_

    def pdg(self):
        """Get the PDG code of the particle."""
        if self.pdg_ == None:
            raise ValueError("pdg not set")
        else:
            return self.pdg_

    def ID(self):
        """Get the ID of the particle. This is a unique number in SMASH."""
        if self.ID_ == None:
            raise ValueError("ID not set")
        else:
            return self.ID_

    def charge(self):
        """Get the electrical charge of the particle."""
        if self.charge_ == None:
            raise ValueError("charge not set")
        else:
            return self.charge_

    def ncoll(self):
        """Get the number of collisions of the particle."""
        if self.ncoll_ == None:
            raise ValueError("ncoll not set")
        else:
            return self.ncoll_

    def form_time(self):
        """Get the formation time of the particle."""
        if self.form_time_ == None:
            raise ValueError("form_time not set")
        else:
            return self.form_time_

    def xsecfac(self):
        """Get the crosssection scaling factor of the particle."""
        if self.xsecfac_ == None:
            raise ValueError("xsecfac not set")
        else:
            return self.xsecfac_

    def proc_id_origin(self):
        """Get the process ID of the particle's origin."""
        if self.proc_id_origin_ == None:
            raise ValueError("proc_id_origin not set")
        else:
            return self.proc_id_origin_

    def proc_type_origin(self):
        """Get the process type of the particle's origin."""
        if self.proc_type_origin_ == None:
            raise ValueError("proc_type_origin not set")
        else:
            return self.proc_type_origin_

    def t_last_coll(self):
        """Get the last time of a collision of the particle."""
        if self.t_last_coll_ == None:
            raise ValueError("t_last_coll not set")
        else:
            return self.t_last_coll_

    def pdg_mother1(self):
        """Get the PDG code of the first mother particle."""
        if self.pdg_mother1_ == None:
            raise ValueError("pdg_mother1 not set")
        else:
            return self.pdg_mother1_

    def pdg_mother2(self):
        """Get the PDG code of the second mother particle."""
        if self.pdg_mother2_ == None:
            raise ValueError("pdg_mother2 not set")
        else:
            return self.pdg_mother2_
        
    def status(self):
        """
        Get the hadron status (stores information on the module origin of 
        a JETSCAPE hadron).
        """
        if self.status_ == None:
            raise ValueError("status not set")
        else:
            return self.status_
        
    def baryon_number(self):
        """Get the baryon number of the particle."""
        if self.baryon_number_ == None:
            raise ValueError("baryon number not set")
        else:
            return self.baryon_number_
    
    def print_particle(self):
        """Print the whole particle information as csv string."""
        print('t,x,y,z,mass,E,px,py,pz,pdg,ID,charge,ncoll,form_time,xsecfac,\
              proc_id_origin,proc_type_origin,t_last_coll,pdg_mother1,\
              pdg_mother2,status,baryon_number')
        print(f'{self.t_},{self.x_},{self.y_},{self.z_},{self.mass_},{self.E_},\
              {self.px_},{self.py_},{self.pz_},{self.pdg_},{self.ID_},\
              {self.charge_},{self.ncoll_},{self.form_time_},{self.xsecfac_},\
              {self.proc_id_origin_},{self.proc_type_origin_}\
              ,{self.t_last_coll_},{self.pdg_mother1_},{self.pdg_mother2_},\
              {self.status_},{self.baryon_number_}')


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
            self.set_t(float(line_from_file[0]))
            self.set_x(float(line_from_file[1]))
            self.set_y(float(line_from_file[2]))
            self.set_z(float(line_from_file[3]))
            self.set_mass(float(line_from_file[4]))
            self.set_E(float(line_from_file[5]))
            self.set_px(float(line_from_file[6]))
            self.set_py(float(line_from_file[7]))
            self.set_pz(float(line_from_file[8]))
            self.set_pdg(int(line_from_file[9]))
            self.set_ID(int(line_from_file[10]))
            self.set_charge(int(line_from_file[11]))
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
              and len(line_from_file)<=21 and len(line_from_file)>=20:
            self.set_t(float(line_from_file[0]))
            self.set_x(float(line_from_file[1]))
            self.set_y(float(line_from_file[2]))
            self.set_z(float(line_from_file[3]))
            self.set_mass(float(line_from_file[4]))
            self.set_E(float(line_from_file[5]))
            self.set_px(float(line_from_file[6]))
            self.set_py(float(line_from_file[7]))
            self.set_pz(float(line_from_file[8]))
            self.set_pdg(int(line_from_file[9]))
            self.set_ID(int(line_from_file[10]))
            self.set_charge(int(line_from_file[11]))
            self.set_ncoll(int(line_from_file[12]))
            self.set_form_time(float(line_from_file[13]))
            self.set_xsecfac(int(line_from_file[14]))
            self.set_proc_id_origin(int(line_from_file[15]))
            self.set_proc_type_origin(int(line_from_file[16]))
            self.set_t_last_coll(float(line_from_file[17]))
            self.set_pdg_mother1(int(line_from_file[18]))
            self.set_pdg_mother2(int(line_from_file[19]))
            if len(line_from_file)==21:
                self.set_baryon_number(int(line_from_file[20]))
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
            self.set_ID(int(line_from_file[0]))
            self.set_pdg(int(line_from_file[1]))
            self.set_status(int(line_from_file[2]))
            self.set_E(float(line_from_file[3]))
            self.set_px(float(line_from_file[4]))
            self.set_py(float(line_from_file[5]))
            self.set_pz(float(line_from_file[6]))

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
        return 0.5 * np.log((self.E() + self.pz()) / (self.E() - self.pz()))
    
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
        if np.abs(self.E()**2. - self.p_abs()**2.) > 1e-6 and\
              self.E()**2. - self.p_abs()**2. > 0.:
            return np.sqrt(self.E()**2.-self.p_abs()**2.)
        else:
            return 0.
        
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
        