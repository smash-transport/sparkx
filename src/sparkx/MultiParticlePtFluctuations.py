#===================================================
#
#    Copyright (c) 2023-2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
#===================================================

import numpy as np


class MultiParticlePtFluctuations:

    def __init__(self, max_order):

        self.max_order = max_order
        # Check if max_order is an integer
        if not isinstance(self.max_order, int):
            raise ValueError("max_order must be an integer")
        # Check that max_order is greater than 0 and less than 9
        if self.max_order < 1 or self.max_order > 8:
            raise ValueError("max_order must be greater than 0 and less than 9")

    def P_W_k(self, particle_list_event):
        Pk = np.zeros(self.max_order)
        Wk = np.zeros(self.max_order)
        for particle in particle_list_event:
            for k in range(1,self.max_order+1):
                Pk[k] += (particle.weight * particle.pt_abs())**k
                Wk[k] += particle.weight**k
        return (Pk, Wk)
    
    def transverse_momentum_correlations_event(self, particle_list_event):
        Pk, Wk = self.P_W_k(particle_list_event, self.max_order)
        
        N = np.zeros(self.max_order)
        D = np.zeros(self.max_order)
        for order in range(self.max_order):
            if order == 0: # k = 1
                N[order] = Pk[order]
                D[order] = Wk[order]
            elif order == 1: # k = 2
                N[order] = Pk[0]**2. - Pk[1]
                D[order] = Wk[0]**2. - Wk[1]
            elif order == 2: # k = 3
                N[order] = Pk[0]**3. - 3.*Pk[1]*Pk[0] + 2.*Pk[2]
                D[order] = Wk[0]**3. - 3.*Wk[1]*Wk[0] + 2.*Wk[2]
            elif order == 3: # k = 4
                N[order] = (Pk[0]**4. - 6.*Pk[1]**2.*Pk[0] + 3.*Pk[1]**2. 
                            + 8.*Pk[2]*Pk[0] - 6.*Pk[3])
                D[order] = (Wk[0]**4. - 6.*Wk[1]**2.*Wk[0] + 3.*Wk[1]**2. 
                            + 8.*Wk[2]*Wk[0] - 6.*Wk[3])
            elif order == 4: # k = 5
                N[order] = (Pk[0]**5. - 10.*Pk[1]*Pk[0]**3. 
                            + 15.*Pk[1]**2.*Pk[0] + 20.*Pk[2]*Pk[0]**2.
                            - 20.*Pk[2]*Pk[1] - 30.*Pk[3]*Pk[0] + 24.*Pk[4])
                D[order] = (Wk[0]**5. - 10.*Wk[1]*Wk[0]**3.
                            + 15.*Wk[1]**2.*Wk[0] + 20.*Wk[2]*Wk[0]**2.
                            - 20.*Wk[2]*Wk[1] - 30.*Wk[3]*Wk[0] + 24.*Wk[4])
            elif order == 5: # k = 6
                N[order] = (Pk[0]**6. - 15.*Pk[1]*Pk[0]**4.
                            + 45.*Pk[0]**2.*Pk[1]**2. - 15.*Pk[1]**3.
                            - 40.*Pk[2]*Pk[0]**3. - 120.*Pk[2]*Pk[1]*Pk[0]
                            + 40.*Pk[2]**2. - 90.*Pk[3]*Pk[0]**2. 
                            + 90.*Pk[3]*Pk[1] + 144.*Pk[4]*Pk[0] - 120.*Pk[5])
                D[order] = (Wk[0]**6. - 15.*Wk[1]*Wk[0]**4.
                            + 45.*Wk[0]**2.*Wk[1]**2. - 15.*Wk[1]**3.
                            - 40.*Wk[2]*Wk[0]**3. - 120.*Wk[2]*Wk[1]*Wk[0]
                            + 40.*Wk[2]**2. - 90.*Wk[3]*Wk[0]**2. 
                            + 90.*Wk[3]*Wk[1] + 144.*Wk[4]*Wk[0] - 120.*Wk[5])
            elif order == 6: # k = 7
                N[order] = (Pk[0]**7. - 21.*Pk[1]*Pk[0]**5.
                            + 105.*Pk[0]**3.*Pk[1]**2. - 105.*Pk[1]**3.*Pk[0]
                            + 70.*Pk[2]*Pk[0]**4. - 420.*Pk[2]*Pk[1]*Pk[0]**2.
                            + 210.*Pk[2]*Pk[1]**2. + 280.*Pk[2]**2.*Pk[0]
                            - 210.*Pk[3]*Pk[0]**3. - 630.*Pk[3]*Pk[1]*Pk[0]
                            - 420.*Pk[3]*Pk[2] + 504.*Pk[4]*Pk[0]**2.
                            - 504.*Pk[4]*Pk[1] - 840.*Pk[5]*Pk[0] + 720.*Pk[6])
                D[order] = (Wk[0]**7. - 21.*Wk[1]*Wk[0]**5.
                            + 105.*Wk[0]**3.*Wk[1]**2. - 105.*Wk[1]**3.*Wk[0]
                            + 70.*Wk[2]*Wk[0]**4. - 420.*Wk[2]*Wk[1]*Wk[0]**2.
                            + 210.*Wk[2]*Wk[1]**2. + 280.*Wk[2]**2.*Wk[0]
                            - 210.*Wk[3]*Wk[0]**3. - 630.*Wk[3]*Wk[1]*Wk[0]
                            - 420.*Wk[3]*Wk[2] + 504.*Wk[4]*Wk[0]**2.
                            - 504.*Wk[4]*Wk[1] - 840.*Wk[5]*Wk[0] + 720.*Wk[6])
            elif order == 7: # k = 8
                N[order] = (Pk[0]**8. - 28.*Pk[1]*Pk[0]**6. 
                            - 210.*Pk[1]**2.*Pk[0]**4. 
                            - 420.*Pk[1]**3.*Pk[0]**2. + 105.*Pk[1]**4.
                            + 112.*Pk[2]*Pk[0]**5. + 1120.*Pk[2]*Pk[1]*Pk[0]**3.
                            + 1680.*Pk[2]*Pk[1]**2.*Pk[0] 
                            + 1120.*Pk[2]**2.*Pk[0]**2. + 1120.*Pk[2]**2.*Pk[1]
                            - 420.*Pk[3]*Pk[0]**4. + 2520.*Pk[3]*Pk[1]*Pk[0]**2.
                            - 1260.*Pk[3]*Pk[1]**2. - 3360.*Pk[3]*Pk[2]*Pk[0]
                            + 1260.*Pk[4]**2. + 1344.*Pk[4]*Pk[0]**3.
                            - 4032.*Pk[4]*Pk[1]*Pk[0] + 2688.*Pk[4]*Pk[2]
                            - 3360.*Pk[5]*Pk[0]**2. + 3360.*Pk[5]*Pk[1]
                            + 5760.*Pk[6]*Pk[0] - 5040.*Pk[7])
                D[order] = (Wk[0]**8. - 28.*Wk[1]*Wk[0]**6.
                            - 210.*Wk[1]**2.*Wk[0]**4.
                            - 420.*Wk[1]**3.*Wk[0]**2. + 105.*Wk[1]**4.
                            + 112.*Wk[2]*Wk[0]**5. + 1120.*Wk[2]*Wk[1]*Wk[0]**3.
                            + 1680.*Wk[2]*Wk[1]**2.*Wk[0]
                            + 1120.*Wk[2]**2.*Wk[0]**2. + 1120.*Wk[2]**2.*Wk[1]
                            - 420.*Wk[3]*Wk[0]**4. + 2520.*Wk[3]*Wk[1]*Wk[0]**2.
                            - 1260.*Wk[3]*Wk[1]**2. - 3360.*Wk[3]*Wk[2]*Wk[0]
                            + 1260.*Wk[4]**2. + 1344.*Wk[4]*Wk[0]**3.
                            - 4032.*Wk[4]*Wk[1]*Wk[0] + 2688.*Wk[4]*Wk[2]
                            - 3360.*Wk[5]*Wk[0]**2. + 3360.*Wk[5]*Wk[1]
                            + 5760.*Wk[6]*Wk[0] - 5040.*Wk[7])

        return N, D
    
    def compute_mean_pt_correlations(self, particle_list_all_events):
        sum_numerator = np.zeros(self.max_order)
        sum_denominator = np.zeros(self.max_order)
        for i in range(len(particle_list_all_events)):
            N, D = self.transverse_momentum_correlations_event(particle_list_all_events[i])
            sum_numerator += D * (N/D)
            sum_denominator += D
        