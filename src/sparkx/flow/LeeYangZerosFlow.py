from sparkx.flow import FlowInterface
import numpy as np
import random as rd
import warnings

rd.seed(42)

class LeeYangZerosFlow(FlowInterface.FlowInterface):
    """
    Compute integrated and transverse momentum dependent anisotropic flow
    using the Lee-Yang zero method from
    - [1] Phys. Lett. B 580 (2004) 157  [nucl-th/0307018]
    - [2] Nucl. Phys. A 727 (2003) 373  [nucl-th/0310016]
    - [3] J. Phys. G: Nucl. Part. Phys. 30 (2004) S1213  [nucl-th/0402053]
    """

    def __init__(self,n=2):

        self.theta_space = np.linspace(0.,np.pi/2,5)
        self.r_space = np.linspace(0.,50,50)
        self.G = np.zeros((len(self.theta_space),len(self.r_space)),dtype=np.complex_)

        if not isinstance(n, int):
            raise TypeError('n has to be int')
        elif n <= 0:
            raise ValueError('n-th harmonic with value n<=0 can not be computed')
        else:
            self.n_ = n

        self.j01_ = 2.4048256
        self.J1rootJ0 = 0.5191147 # J1(j01)


    def __g_theta(self, n, r, theta, weight_j, phi_j):
        if not isinstance(weight_j, (list, np.ndarray)) or not isinstance(phi_j, (list, np.ndarray)):
            raise ValueError("Not the correct input format for g_theta")
        if len(weight_j) != len(phi_j):
            raise ValueError("weight_j and phi_j do not have the same length")

        g_theta = 1.0 + 0.0j
        for j in range(len(weight_j)):
            g_theta *= (1.0 + 1.0j * r * weight_j[j] * np.cos(n * (phi_j[j] - theta)))
        return g_theta

    def __Q_x(self, n, weight_j, phi_j):
        if not isinstance(weight_j,(list,np.ndarray)) or not isinstance(phi_j,(list,np.ndarray)):
            raise ValueError('Not the correct input format for g_theta')
        if len(weight_j) != len(phi_j):
            raise ValueError('weight_j and phi_j do not have the same length')
        Q_x = 0.
        for j in range(len(weight_j)):
            Q_x += weight_j[j] * np.cos(n * phi_j[j])
        return Q_x

    def __Q_y(self,n,weight_j,phi_j):
        if not isinstance(weight_j,(list,np.ndarray)) or not isinstance(phi_j,(list,np.ndarray)):
            raise ValueError('Not the correct input format for g_theta')
        if len(weight_j) != len(phi_j):
            raise ValueError('weight_j and phi_j do not have the same length')
        Q_y = 0.
        for j in range(len(weight_j)):
            Q_y += weight_j[j] * np.sin(n * phi_j[j])
        return Q_y

    def sigma(self,QxSqPQySqAll,QxAll,QyAll,VnInfty):
        # Eq. 7 in Ref. [3]
        return QxSqPQySqAll - QxAll*QxAll - QyAll*QyAll - VnInfty*VnInfty

    def chi(self,VnInfty,sigma):
        return VnInfty / sigma

    def relative_Vn_fluctuation(self,NEvents,chi):
        # Eq. 8 in Ref. [3]
        return (1./(2.*NEvents*self.j01**2.*self.J1rootJ0**2.)) * (np.exp(self.j01**2. / (2.*chi*chi)) + np.exp(-self.j01**2. / (2.*chi**2.)) * (-0.2375362))

    
    def integrated_flow(self, particle_data):
        QxAll = 0.
        QyAll = 0.
        QxSqPQySqAll = 0.
        number_events = len(particle_data)

        for event in range(number_events):
            phi_j = []
            weight_j = []
            event_multiplicity = len(particle_data[event])

            print("event = ",event,", mult = ",len(particle_data[event]))
            for particle in particle_data[event]:
                phi_j.append(particle.phi())
                weight_j.append(1./event_multiplicity)

            # randomize the event plane
            rand_event_plane = rd.uniform(0.,2.*np.pi)
            phi_j = [phi+rand_event_plane for phi in phi_j]

            Qx = self.__Q_x(2.,weight_j,phi_j)
            Qy = self.__Q_y(2.,weight_j,phi_j)
            QxSqPQySqAll += (Qx**2. + Qy**2.) / event_multiplicity

            g = np.zeros((len(self.theta_space),len(self.r_space)),dtype=np.complex_)
            for theta in range(len(self.theta_space)):
                for r in range(len(self.r_space)):
                    g[theta][r] = self.__g_theta(2,self.r_space[r],self.theta_space[theta],weight_j,phi_j)
                    self.G[theta][r] += g[theta][r] / number_events

        min_r0_theta = []
        for theta in range(len(self.theta_space)):
            min_idx = np.where(abs(self.G[theta]) == abs(self.G[theta]).min())
            min_r0_theta.append(self.r_space[min_idx])
            if min_idx == len(self.r_space)-1:
                warnings.warn("The minimum is at the border of the selected r \
                              range. Please choose a larger r.")
        
        j01 = 2.40483
        vn_inf_theta = []
        for theta in range(len(self.theta_space)):
            vn_inf_theta.append(j01 / min_r0_theta[theta])

        vn_inf = np.mean(vn_inf_theta)
        sigma_value = self.sigma(QxSqPQySqAll,QxAll,QyAll,vn_inf)
        chi_value = self.chi(vn_inf,sigma_value)
        # factor 1/2 because the statistical error on Vninf is a faactor 2 smaller
        relative_Vn_fluctuation = (1./2)*self.relative_Vn_fluctuation(number_events,chi_value)
        
        return [vn_inf,sigma_value,chi_value,relative_Vn_fluctuation]







    def differential_flow(self,particle_data,bins,flow_as_function_of):

        return None