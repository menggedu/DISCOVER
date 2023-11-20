import numpy as np
import pandas as pd
import scipy.io as scio
from dso.task.pde.pde import PDETask
from dso.library import Library
from dso.functions import create_tokens


class PDE_dsr(PDETask):
    """
    Class for the symbolic regression task. 
    """

    task_type = "pde_dsr"


    def reward_function(self,p):
        y_hat= p.execute(self.u, self.x, self.ut)
    
        if p.invalid:

            return self.invalid_reward

        # Compute metric
        r = self.metric(self.ut, y_hat)

        return r

    def evaluate(self, p):

        # Compute predictions on test data
        y_hat= p.execute(self.u, self.x, self.ut)


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

        return info




if __name__ == '__main__':
    pass