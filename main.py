import cvxpy as cp
import numpy as np


class Benders:
    """
    Solves MILPs of the following form using Bender's decomposition:

    min c_T*x + f_T*y     over x,y
    s.t   A*x + B*y >= b (m constraints)
          D*y >= d       (n constraints)
          x >= 0         (Nx-dimensional vector)
          y >= 0         (Ny-dimensional vector)
          y integer
    """
    def __init__(self):
        # Data
        self.Nx = 2  # Size of vector x
        self.Ny = 1  # Size of vector y
        self.m = 2  # Number of complicating constraints A*x + B*y >= b
        self.n = 1  # Number of constraints D*y >= d
        self.c = np.array([-4, -3]).reshape((self.Nx, 1))
        self.f = np.array([-5]).reshape((self.Ny, 1))
        self.A = np.array([[-2, -3], [-2, -1]]).reshape((self.m, self.Nx))
        self.B = np.array([-1, -3]).reshape((self.m, self.Ny))
        self.b = np.array([-12, -12]).reshape((self.m, 1))
        self.D = np.array([-1]).reshape((self.n, self.Ny))
        self.d = np.array([-20]).reshape((self.n, 1))

        self.y_init = np.array([0]).reshape((self.Ny, 1))  # Initial feasible guess

        self.eps = 1e-6              # Convergence value
        self.max_iterations = 30  # Number of maximum iterations

        self.LB = float('-inf')  # Lower bound of objective function
        self.UB = float('inf')   # Upper bound of objective function
        self.optimality_cuts = []
        self.feasibility_cuts = []
        self.lower_bounds = []
        self.upper_bounds = []

    def solve_problem(self):
        """ """
        i = 0
        y_curr = self.y_init
        while self.UB - self.LB >= self.eps and i <= self.max_iterations:

            # Solve sub-problem
            p, obj_value_sp, sp_status, x_sol = self.solve_subproblem(y_curr)

            # Add optimality or feasibility cut for Master problem
            if sp_status == 'optimal':  # Sub-problem feasible
                # Add optimality cut
                self.optimality_cuts.append(p)

                # Update upper bound
                self.UB = min(self.UB, np.dot(np.transpose(self.f), y_curr).item() + obj_value_sp)

            else:                       # Sub-problem unbounded
                # Solve Modified Sub-problem
                r = self.solve_modified_subproblem(y_curr)

                # Add feasibility cut
                self.feasibility_cuts.append(r)

            # Solve Master problem
            y_curr, obj_value_master = self.solve_master_problem()

            # Update lower bound
            self.LB = obj_value_master

            i = i + 1
            # Save values for plotting
            self.lower_bounds.append(self.LB)
            self.upper_bounds.append(self.UB)

            print("Iteration i={} : UB={}, LB={} ".format(i, self.UB, self.LB))
        
        _, _, _, x_sol = self.solve_subproblem(y_curr)

        if i > self.max_iterations:
            print("\nThe algorithm did not converge in the given iterations.")
        else:
            print("\n*** Optimal solution to the MILP problem found. ***")
        print("The optimal value is: {}".format(obj_value_master))
        print("The optimal solution is x*={}, y*={}".format(*x_sol, *y_curr))

    def solve_subproblem(self, y):
        """ """
        p = cp.Variable((self.m, 1))
        objective = p.T@(self.b-self.B@y)
        constraints = [p.T@self.A <= self.c.T, p >= np.zeros((self.m, 1))]
        prob = cp.Problem(cp.Maximize(objective), constraints)
        prob.solve(solver='MOSEK', verbose=False)

        x_sol = constraints[0].dual_value

        return p.value, prob.value, prob.status, x_sol

    def solve_modified_subproblem(self, y):
        """ """
        p = cp.Variable((self.m, 1))
        constraints = [p.T@(self.b-self.B@y) == 1, p.T@self.A <= np.zeros((1, self.Nx)), p >= np.zeros((self.m, 1))]
        prob = cp.Problem(cp.Maximize(0), constraints)
        prob.solve(solver='MOSEK', verbose=False)

        return p.value

    def solve_master_problem(self):
        """"""
        y = cp.Variable((self.Ny, 1), integer=True)
        n = cp.Variable(1)
        objective = self.f.T@y + n
        constraints = [self.D@y >= self.d, y >= np.zeros((self.Ny, 1))]
        for p in self.optimality_cuts:
            constraints.append(n >= p.T@(self.b-self.B@y))
        for r in self.feasibility_cuts:
            constraints.append(0 >= r.T@(self.b-self.B@y))
        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver='MOSEK')

        return y.value, prob.value


if __name__ == '__main__':

    benders_dec = Benders()
    benders_dec.solve_problem()
