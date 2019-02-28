import time
import numpy as np
from numpy.linalg import eigh

from ase.optimize.optimize import Optimizer


# An ASE-like optimizer for constrained minimization
class PCOBFGS(Optimizer):
    """Projected Constrained Optimization optimizer.

    A quasi-Newton geometry minimization algorithm for use with bond
    distance, bond angles, and dihedral angle constraints.

    """
    def __init__(self, atoms, restart=None, logfile='-',
                 trajectory=None, master=None, force_consistent=None,
                 bonds=None, angles=None, dihedrals=None, Rtrust=0.04,
                 Rmax=0.5, dec_lb=0.5, dec_ub=1.5, inc_lb=0.75, inc_ub=1.25,
                 Sf=2., sigma_angle=1., sigma_dihedral=0.25):
        """Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        restart: string
            Currently not used.

        logfile: file object or str
            if *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        trajectory: string
            ASE Trajectory file used to store trajectory of atomic movement.

        master: boolean
            Defaults to None, which causes only rank 0 to save files.
            If set to True, this rank will save files.

        force_consistent: boolean or None
            Use force-consistent energy calls (as opposed to the energy
            extrapolated to 0 K). By default (force_consistent=None),
            uses force-consistent energies if available in the
            calculator, but falls back to force_consistent=False if not.

        bonds: iterable of tuples or None
            A specification of which interatomic distances should be
            constrained during optimization. Entries should either have
            the form:
            `(val, (idx1, idx2))`
            to constrain the distance between the atoms with indices
            `idx1` and `idx2` to a value of `val`, or the form:
            `(idx1, idx2)`
            to constrain the distance between these atoms to its
            current value.

        angles: iterable of tuples or None
            A specification of which angles should be constrained
            during optimization. Entries can either have the form:
            `(val, (idx1, idx2, idx3))`
            or the form:
            `(idx1, idx2, idx3)`.
            See the description of `bonds` for more details.

        dihedrals: iterable of tuples or None
            A specification of which dihedral angles should be
            constrained during optimization. Entires can either
            have the form:
            `(val, (idx1, idx2, idx3, idx4))`
            or the form:
            `(idx1, idx2, idx3, idx4)`.
            See the description of `bonds` for more details.

        Rtrust: float
            Initial trust radius for geometry optimization. This value
            will change over the course of optimization.

        Rmax: float
            Maximum trust radius. The trust radius will not increase
            this value.

        dec_lb: float
            Lower bound for decreasing the trust radius. If the ratio
            of the true change in energy to the expected change in
            energy is less than this value, the trust radius will be
            decreased.

        dec_ub: float
            Upper bound for decreasing the trust radius. If the ratio
            described in `dec_lb` is above this value, the trust radius
            will be increased.

        inc_lb: float
            Lower bound for increasing the trust radius. If the ratio
            described in `dec_lb` is between this value and `inc_ub`
            and the magnitude of the lastest step was equal to `Rtrust`,
            then the trust radius will be increased.

        inc_ub: float
            Upper bound for increasing the trust radius. See `inc_lb`.

        Sf: float
            Scaling factor for increasing/decreasing the trust radius.
            The trust radius will be decreased by a factor of `Sf`,
            and increased by a factor of `sqrt(Sf)`.

        sigma_angle: float
            Weight for angle constraints. Affects the rate at which
            angle constraints are satisfied. If you experience
            difficulties achieving convergence, try decreasing this
            value from its default value of `1.`.

        sigma_dihedral: float
            Weight for dihedral constraints. See `sigma_angle` for
            more details.

        """
        Optimizer.__init__(self, atoms, restart, logfile, trajectory, master,
                           force_consistent=force_consistent)
        self.function_calls = 0
        self.force_calls = 0

        pos = atoms.get_positions()

        self.bonds = []
        if bonds is not None:
            for bond in bonds:
                if isinstance(bond[1], int):
                    target = self.atoms.get_distance(*bond)
                    self.bonds.append((target, bond))
                else:
                    self.bonds.append(bond)

        self.angles = []
        if angles is not None:
            for angle in angles:
                if isinstance(angle[1], int):
                    i, j, k = angle
                    r12 = pos[j] - pos[i]
                    r23 = pos[k] - pos[j]
                    d12 = np.linalg.norm(r12)
                    d23 = np.linalg.norm(r23)
                    cosq = np.dot(r12, r23) / (d12 * d23)
                    target = np.arccos(cosq)
                    self.angles.append((target, angle))
                else:
                    self.angles.append(angle)

        self.dihedrals = []
        if dihedrals is not None:
            for dihedral in dihedrals:
                if isinstance(dihedral[1], int):
                    i1, i2, i3, i4 = dihedral
                    r12 = pos[i2] - pos[i1]
                    r23 = pos[i3] - pos[i2]
                    r34 = pos[i4] - pos[i3]

                    d23 = np.linalg.norm(r23)
                    r12xr23 = np.cross(r12, r23)
                    r23xr34 = np.cross(r23, r34)

                    numer = np.dot(r23, np.cross(r12xr23, r23xr34)) / d23
                    denom = np.dot(r12xr23, r23xr34)
                    target = np.arctan2(numer, denom)
                    self.dihedrals.append((target, dihedral))
                else:
                    self.dihedrals.append(dihedral)

        # Number of constraints
        self.nbonds = len(self.bonds)
        self.nangles = len(self.angles)
        self.ndihedrals = len(self.dihedrals)
        self.m = self.nbonds + self.nangles + self.ndihedrals

        self.d = 3 * len(self.atoms)
        self.W = np.eye(self.d) * 70
        self.Rmax = Rmax
        self.Rtrust = Rtrust
        self.dec_lb = dec_lb
        self.dec_ub = dec_ub
        self.inc_lb = inc_lb
        self.inc_ub = inc_ub
        self.Sf = Sf
        self.sigma_angle = sigma_angle
        self.sigma_dihedral = sigma_dihedral

        self.hlast = None
        self.xlast = None
        self.Tc = None
        self.Tm = None
        self.dq = None
        self.dq_x = None
        self.dq_y = None
        self.g = None

    def r(self, x):
        """Calculate the error and Jacobian for all constraints"""
        r = np.zeros(self.m)
        drdx = np.zeros((self.m, self.d))

        r_bonds, drdx_bonds = self._r_bonds(x)
        r[:self.nbonds] = r_bonds
        drdx[:self.nbonds] = drdx_bonds
        nconstr = self.nbonds

        r_angles, drdx_angles = self._r_angles(x)
        r[nconstr:nconstr+self.nangles] = r_angles
        drdx[nconstr:nconstr+self.nangles] = drdx_angles
        nconstr += self.nangles

        r_dihedrals, drdx_dihedrals = self._r_dihedrals(x)
        r[nconstr:nconstr+self.ndihedrals] = r_dihedrals
        drdx[nconstr:nconstr+self.ndihedrals] = drdx_dihedrals

        return r, drdx

    def _r_bonds(self, x):
        """Calculate the error and Jacobian of bond distance constraints"""
        r_bonds = np.zeros(self.nbonds)
        drdx_bonds = np.zeros((self.nbonds, self.d))
        nconstr = 0
        for target, (i, j) in self.bonds:
            r12 = x[j] - x[i]
            d12 = np.linalg.norm(r12)
            dbonddx = r12 / d12
            r_bonds[nconstr] = d12 - target
            drdx_bonds[nconstr, 3*i:3*(i+1)] = -dbonddx
            drdx_bonds[nconstr, 3*j:3*(j+1)] = dbonddx
            nconstr += 1
        return r_bonds, drdx_bonds

    def _r_angles(self, x):
        """Calculate the error and Jacobian of angle constraints"""
        r_angles = np.zeros(self.nangles)
        drdx_angles = np.zeros((self.nangles, self.d))
        nconstr = 0
        for target, (i, j, k) in self.angles:
            r12 = x[j] - x[i]
            r23 = x[k] - x[j]
            d12 = np.linalg.norm(r12)
            d23 = np.linalg.norm(r23)
            cosq = np.dot(r12, r23) / (d12 * d23)
            q = np.arccos(cosq)
            r_angles[nconstr] = q - target

            sinq = np.sqrt(1 - cosq**2)
            dcosq = np.zeros(9)
            dcosq[:3] = r12 * cosq / d12**2 - r23 / (d12 * d23)
            dcosq[6:] = r12 / (d12 * d23) - r23 * cosq / d23**2
            dcosq[3:6] = -dcosq[:3] - dcosq[6:]
            dqdx = -dcosq/sinq

            drdx_angles[nconstr, 3*i:3*(i+1)] = dqdx[:3]
            drdx_angles[nconstr, 3*j:3*(j+1)] = dqdx[3:6]
            drdx_angles[nconstr, 3*k:3*(k+1)] = dqdx[6:]
            nconstr += 1
        r_angles *= self.sigma_angle
        drdx_angles *= self.sigma_angle
        return r_angles, drdx_angles

    def _r_dihedrals(self, x):
        """Calculate error and Jacobian of dihedral constraints"""
        r_dihedrals = np.zeros(self.ndihedrals)
        drdx_dihedrals = np.zeros((self.ndihedrals, self.d))
        nconstr = 0
        for target, (i, j, k, l) in self.dihedrals:
            r12 = x[j] - x[i]
            r23 = x[k] - x[j]
            r34 = x[l] - x[k]

            d23 = np.linalg.norm(r23)

            r12xr23 = np.cross(r12, r23)
            r23xr34 = np.cross(r23, r34)

            numer = np.dot(r23, np.cross(r12xr23, r23xr34)) / d23
            denom = np.dot(r12xr23, r23xr34)
            q = np.arctan2(numer, denom)

            rq = (q - target + np.pi) % (2 * np.pi) - np.pi
            r_dihedrals[nconstr] = rq

            r12xr34 = np.cross(r12, r34)

            ddenom = np.zeros(12)
            ddenom[:3] = np.cross(r23xr34, r23)
            ddenom[3:6] = np.cross(r12, r23xr34) + np.cross(r12xr23, r34)
            ddenom[6:9] = np.cross(r23, r12xr23)

            ddenom[9:] -= ddenom[6:9]
            ddenom[6:9] -= ddenom[3:6]
            ddenom[3:6] -= ddenom[:3]

            dnumer = np.zeros(12)
            dnumer[:3] = -d23 * r23xr34
            dnumer[3:6] = -r23 * np.dot(r12, r23xr34) / d23 + d23 * r12xr34
            dnumer[6:9] = -d23 * r12xr23

            dnumer[9:] -= dnumer[6:9]
            dnumer[6:9] -= dnumer[3:6]
            dnumer[3:6] -= dnumer[:3]
            dq = (dnumer * denom - numer * ddenom) / (numer**2 + denom**2)

            drdx_dihedrals[nconstr, 3*i:3*(i+1)] = dq[:3]
            drdx_dihedrals[nconstr, 3*j:3*(j+1)] = dq[3:6]
            drdx_dihedrals[nconstr, 3*k:3*(k+1)] = dq[6:9]
            drdx_dihedrals[nconstr, 3*l:3*(l+1)] = dq[9:]
            nconstr += 1
        r_dihedrals *= self.sigma_dihedral
        drdx_dihedrals *= self.sigma_dihedral
        return r_dihedrals, drdx_dihedrals

    def get_basis(self, x):
        """Generate a complete basis to separate constraint degrees of
        freedom from minimization degrees of freedom"""
        r, drdx = self.r(x)
        self.Tc = np.zeros((self.d, self.m))

        for n in range(self.m):
            self.Tc[:, n] = drdx[n] / np.linalg.norm(drdx[n])
            while True:
                for k in range(n):
                    self.Tc[:, n] -= (self.Tc[:, k]
                                      * np.dot(self.Tc[:, n], self.Tc[:, k]))
                Tcn_norm = np.linalg.norm(self.Tc[:, n])
                self.Tc[:, n] /= Tcn_norm
                if np.abs(1. - Tcn_norm) < 1e-13:
                    break
        nTm = 0
        self.Tm = np.zeros((self.d, self.d - self.m))
        for i in range(self.d):
            u = np.zeros(self.d)
            u[i] = 1.
            while True:
                for n in range(self.m):
                    u -= self.Tc[:, n] * np.dot(self.Tc[:, n], u)
                for j in range(nTm):
                    u -= self.Tm[:, j] * np.dot(self.Tm[:, j], u)
                u_norm = np.linalg.norm(u)
                if u_norm < 1e-3:
                    break
                u /= u_norm
                if np.abs(1. - u_norm) < 1e-13:
                    self.Tm[:, nTm] = u
                    nTm += 1
                    break
            if nTm == self.d - self.m:
                break
        else:
            raise RuntimeError('Gram-Schmidt failed!')

    def step(self, f):
        x = self.atoms.get_positions()
        r, drdx = self.r(x)
        e = self.atoms.get_potential_energy(
                force_consistent=self.force_consistent)

        if self.dq is not None:
            Q = (np.dot(self.dq, self.glast)
                 - np.einsum('i,ij,j', self.dq_y, self.W, self.dq_x)
                 + 0.5 * np.einsum('i,ij,j', self.dq_y, self.W, self.dq_y)
                 + 0.5 * np.einsum('i,ij,j', self.dq_x, self.W, self.dq_x))
            ratio = (e - self.elast) / Q
            if ratio < self.dec_lb or ratio > self.dec_ub:
                self.Rtrust /= self.Sf
            elif (self.inc_lb < ratio < self.inc_ub
                    and abs(np.linalg.norm(self.dq_x) - self.Rtrust) < 1e-8):
                self.Rtrust *= np.sqrt(self.Sf)
            self.Rtrust = min(self.Rtrust, self.Rmax)

        g = -f.ravel()
        self.get_basis(x)
        h = g - np.dot(np.dot(self.Tc.T, g), drdx)
        if self.hlast is not None:
            dh = h - self.hlast
            dx = x.ravel() - self.xlast.ravel()
            self.update_H(dx, dh)
        self.hlast = h.copy()
        self.xlast = x.copy()

        TWT = np.dot(self.Tm.T, np.dot(self.W, self.Tm))
        TWTlams, TWTvecs = eigh(TWT)
        g_red = (np.dot(self.Tm.T, g)
                 - np.dot(self.Tm.T, np.dot(self.W, np.dot(self.Tc, r))))
        L = np.abs(TWTlams)
        Vg = np.dot(TWTvecs.T, g_red)
        dx = -np.dot(TWTvecs, Vg / L)
        dx_mag = np.linalg.norm(dx)
        xi = 0.5
        if dx_mag > self.Rtrust:
            xilower = 0
            xiupper = None
            while True:
                if xiupper is None:
                    xi *= 2
                else:
                    xi = (xilower + xiupper) / 2.
                dx = -np.dot(TWTvecs, Vg / (L + xi))
                dx_mag = np.linalg.norm(dx)
                if abs(dx_mag - self.Rtrust) < 1e-14 * self.Rtrust:
                    break

                if dx_mag > self.Rtrust:
                    xilower = xi
                else:
                    xiupper = xi

        self.dq_x = np.dot(self.Tm, dx)
        self.dq_y = -np.dot(self.Tc, r)
        self.dq = self.dq_x + self.dq_y
        self.glast = g
        self.elast = e
        self.atoms.set_positions(x + self.dq.reshape((-1, 3)))

    # TS-BFGS update
    def update_H(self, s, y):
        """Update the modified Hessian matrix using the TS-BFGS algorithm"""
        Wlams, Wvecs = eigh(self.W)
        j = y - np.dot(self.W, s)
        x1 = y * np.dot(s.T, y)
        absWs = np.dot(Wvecs, np.abs(Wlams) * np.dot(Wvecs.T, s))
        x2 = absWs * np.dot(s.T, absWs)
        u = (x1 + x2) / np.dot(x1 + x2, s)
        ujT = np.outer(u, j)
        self.W += (ujT + ujT.T) - np.dot(np.outer(u, j), np.outer(s, u))

    def converged(self, forces=None):
        if forces is None:
            forces = self.atoms.get_forces()
        if self.Tm is None:
            self.get_basis(self.atoms.get_positions())
        r, drdx = self.r(self.atoms.get_positions())
        gproj = -np.dot(self.Tm,
                        np.dot(self.Tm.T, forces.ravel())).reshape((-1, 3))
        gmax = (gproj**2).sum(axis=1).max()
        return gmax < self.fmax**2 and np.linalg.norm(r) < self.fmax

    def log(self, forces):
        """Logging modified to print projected gradient"""
        if self.Tm is None:
            self.get_basis(self.atoms.get_positions())
        fproj = np.dot(self.Tm,
                       np.dot(self.Tm.T, forces.ravel())).reshape((-1, 3))
        fmax = np.sqrt((fproj**2).sum(axis=1).max())
        e = self.atoms.get_potential_energy(
                force_consistent=self.force_consistent)
        T = time.localtime()
        if self.logfile is not None:
            name = self.__class__.__name__
            if self.nsteps == 0:
                self.logfile.write(
                    '%s  %4s %8s %15s %12s\n' %
                    (' ' * len(name), 'Step', 'Time', 'Energy', 'fmax'))
                if self.force_consistent:
                    self.logfile.write(
                        '*Force-consistent energies used in optimization.\n')
            self.logfile.write('%s:  %3d %02d:%02d:%02d %15.6f%1s %12.4f\n' %
                               (name, self.nsteps, T[3], T[4], T[5], e,
                                {1: '*', 0: ''}[self.force_consistent], fmax))
            self.logfile.flush()
