import numpy as np
from scipy.special.orthogonal import p_roots
from numpy.polynomial.legendre  import leggauss

import pandas as pd
import scipy.io as scio
import math

from dso.task.pde.pde import PDETask



class PolyTask(PDETask):
    """
    Class for the symbolic regression task. Discrete objects are expressions,
    which are evaluated based on their fitness to a specified dataset.
    """

    task_type = "pde_chem"
    model = None
    def __init__(self, 
                ):
        super().__init__(*args, **kwargs)
        import pdb;pdb.set_trace()    
        if test_list is not None and test_list[0] is not None:
            self.u_test,self.ut_test = test_list
            self.ut_test = self.ut_test.reshape(-1,1)
        else:
            self.u_test,self.ut_test = None,None

    def evaluate_new(self, p):
        # Compute predictions on test data
        y_hat,y_right,  w = p.execute(self.u, self.x, self.ut)

        n = len(w)
        # y_hat = p.execute(self.X_test)
        if p.invalid:
            nmse_test = None
            nmse_test_noiseless = None
            success = False

        else:
            # NMSE on test data (used to report final error)
            nmse_test = np.mean((self.ut - y_hat) ** 2)

            # NMSE on noiseless test data (used to determine recovery)
            # nmse_test_noiseless = np.mean((self.y_test_noiseless - y_hat) ** 2) / self.var_y_test_noiseless

            # Success is defined by NMSE on noiseless test data below a threshold
            success = nmse_test < self.threshold

        info = {
            "nmse_test" : nmse_test,
            "success" : success,

        }
        if self.u_test is not None:
        
            y_hat_test,y_right, w_test = p.execute(self.u_test, self.x, self.ut_test, test=True)
            info.update({
                'w_test': w_test
            })

        if self.metric_test is not None:
            if p.invalid:
                m_test = None
                m_test_noiseless = None
            else:
                m_test = self.metric_test(self.y_test, y_hat,n)
                m_test_noiseless = self.metric_test(self.y_test_noiseless, y_hat,n)

            info.update({
                self.extra_metric_test : m_test,
                self.extra_metric_test + '_noiseless' : m_test_noiseless
            })

        return info
    