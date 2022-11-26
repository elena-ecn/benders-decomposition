"""
Solves MILPs using the Benders decomposition method.

Author: Elena Oikonomou
Date: Fall 2022
"""

import cvxpy as cp
import numpy as np
import sys


class Benders:
    """
    Solves MILPs of the following form using Benders decomposition:

    min c_T*x + f_T*y     over x,y
    s.t   A*x + B*y >= b (m constraints)
          D*y >= d       (n constraints)
          x >= 0         (Nx-dimensional vector)
          y >= 0         (Ny-dimensional vector)
          y integer
    """
    def __init__(self):
        # Data
        self.Nx = 2                                        # Size of vector x
        self.Ny = 1                                        # Size of vector y
        self.m = 2                                         # Number of complicating constraints A*x + B*y >= b
        self.n = 1                                         # Number of constraints D*y >= d
        self.c = np.array([-4, -3]).reshape((self.Nx, 1))
        self.f = np.array([-5]).reshape((self.Ny, 1))
        self.A = np.array([[-2, -3], [-2, -1]]).reshape((self.m, self.Nx))
        self.B = np.array([-1, -3]).reshape((self.m, self.Ny))
        self.b = np.array([-12, -12]).reshape((self.m, 1))
        self.D = np.array([-1]).reshape((self.n, self.Ny))
        self.d = np.array([-20]).reshape((self.n, 1))

        self.y_init = np.array([4], dtype=int).reshape((self.Ny, 1))  # Initial feasible guess (Important!)

        self.eps = 1e-3                                    # Convergence value
        self.max_iterations = 20                           # Number of maximum iterations
        self.LB = sys.float_info.min                       # Lower bound of objective function
        self.UB = sys.float_info.max                       # Upper bound of objective function
        self.optimality_cuts = []
        self.feasibility_cuts = []
        self.lower_bounds = []
        self.upper_bounds = []

    def solve_problem(self):
        """Solves the MILP using Benders decomposition."""
        i = 0
        y_sol = self.y_init

        while abs((self.UB - self.LB)/self.UB) >= self.eps and i <= self.max_iterations:

            # Solve sub-problem
            p, obj_value_sp, sp_status, _ = self.solve_subproblem(y_sol)

            # Add optimality or feasibility cut for Master problem
            if sp_status == 'optimal':  # Sub-problem feasible
                # Add optimality cut
                self.optimality_cuts.append(p)

                # Update upper bound
                self.UB = min(self.UB, np.dot(np.transpose(self.f), y_sol).item() + obj_value_sp)

            else:                       # Sub-problem unbounded
                # Solve Modified Sub-problem
                r = self.solve_modified_subproblem(y_sol)

                # Add feasibility cut
                self.feasibility_cuts.append(r)

            # Solve Master problem
            y_sol, obj_value_master = self.solve_master_problem()

            # Update lower bound
            self.LB = obj_value_master

            # Update iteration index
            i = i + 1

            # Save values for plotting
            self.lower_bounds.append(self.LB)
            self.upper_bounds.append(self.UB)

            print("Iteration i={} : UB={}, LB={}".format(i, self.UB, self.LB))

        # Solve sub-problem with the optimal y solution to get the optimal x solution
        _, _, _, x_sol = self.solve_subproblem(y_sol)

        # Display the results
        self.show_results(i, obj_value_master, x_sol, y_sol)

    def show_results(self, i, obj_value_master, x_sol, y_sol):
        """Displays the results of the optimization problem.

        Inputs:
          - i(int):                  The index of last iteration
          - obj_value_master(float): The objective function value of the master problem
          - x_sol(np.ndarray):       The optimal x solution [Nx,1]
          - y_sol(np.ndarray):       The optimal y solution [Ny,1]
        """

        if i > self.max_iterations:
            print("\nThe algorithm did not converge in the given iterations.")
        else:
            print("\n*** Optimal solution to the MILP problem found. ***")
        print("The optimal value is: {}".format(obj_value_master))

        if x_sol is not None:
            print("The optimal solution is x*={}, y*={}".format(*x_sol, *y_sol))
        else:
            print("\nThe algorithm did not find the optimal solution. Please try another initial feasible guess y_init!")

    def solve_subproblem(self, y):
        """Solves the sub-problem in the dual form.

        max π_T*(b-By) over π  (where π is the dual var for the Ax>=b-By constraints)
        s.t.  π_T*A <= c_T
              π >= 0

        Inputs:
          - y(np.ndarray):       The current y solution [Ny,1]
        Returns:
          - p.value(np.ndarray): The optimal dual var p [m,1]
          - prob.value(float):   The optimal objective function value of the sub-problem
          - prob.status(str):    The exit status of the sub-problem
          - x_sol(np.ndarray):   The optimal x solution [Nx,1]
        """
        p = cp.Variable((self.m, 1))
        objective = p.T@(self.b-self.B@y)
        constraints = [p.T@self.A <= self.c.T, p >= np.zeros((self.m, 1))]
        prob = cp.Problem(cp.Maximize(objective), constraints)
        prob.solve(solver='MOSEK', verbose=False)
        x_sol = constraints[0].dual_value  # The dual var of the dual sub-problem is the original var x

        return p.value, prob.value, prob.status, x_sol

    def solve_modified_subproblem(self, y):
        """Solves the modified sub-problem in the dual form to find the extreme ray.

        max 0 over π  (where π is the dual var for the Ax>=b-By constraints)
        s.t.  π_T*(b-By) = 1
              π_T*A <= 0
              π >= 0

        Inputs:
          - y(np.ndarray):       The current y solution [Ny,1]
        Returns:
          - p.value(np.ndarray): The optimal dual var p (extreme ray) [m,1]
        """
        p = cp.Variable((self.m, 1))
        constraints = [p.T@(self.b-self.B@y) == 1, p.T@self.A <= np.zeros((1, self.Nx)), p >= np.zeros((self.m, 1))]
        prob = cp.Problem(cp.Maximize(0), constraints)
        prob.solve(solver='MOSEK', verbose=False)
        return p.value

    def solve_master_problem(self):
        """Solves the Master problem.

        min f_T*y + n   over y,n
        s.t.  n >= πe_T*(b-By)   (optimality cuts)
              0 >= rq_T*(b-By)   (feasibility cuts)
              D*y >= d
              y >= 0
              y integer

        Returns:
          - y.value(np.ndarray): The optimal y solution [Ny,1]
          - prob.value(float):   The optimal objective function value of the Master problem
        """
        y = cp.Variable((self.Ny, 1), integer=True)
        n = cp.Variable(1)
        objective = self.f.T@y + n
        constraints = [self.D@y >= self.d, y >= np.zeros((self.Ny, 1))]
        for p in self.optimality_cuts:
            constraints.append(n >= p.T@(self.b-self.B@y))  # Add optimality cuts
        for r in self.feasibility_cuts:
            constraints.append(0 >= r.T@(self.b-self.B@y))  # Add feasibility cuts
        prob = cp.Problem(cp.Minimize(objective), constraints)

        try:
            prob.solve(solver='MOSEK', verbose=False)
        except cp.SolverError:
            err_msg = "Solver for Master problem failed. Please try a feasible initial solution guess y_init!\n" \
                      "For more info, use verbose=True."
            raise cp.SolverError(err_msg)

        return y.value, prob.value
