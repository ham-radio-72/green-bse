#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    Main script used for solving the dynamic BSE Casida equations            #
#    on the Matsubara frequency grid.                                         #
#                                                                             #

import time
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple
import casidaEq as casida
import h5py
import contract as ct
import numpy as np
import qp
import plasPole

try:
    import psutil
    import multiprocessing as mp
    from joblib import effective_n_jobs
    MONITORING_AVAILABLE = True
except ImportError:
    print("Warning: psutil or joblib not available. System monitoring will be limited.")
    MONITORING_AVAILABLE = False

AU2EV = 27.211386245981


@dataclass
class BSEConfig:
    """Configuration class for BSE calculations."""
    # File paths
    input_file: str = "input.h5"
    sim_file: str = "sim.h5"
    output_file: str = "bse.h5"
    int_path: str = "df_hf_int/"
    ir_file: Optional[str] = None
    pi_file: str = "p_iw_tilde_q0.h5"
    
    # Physical parameters
    beta: float = 1000.0
    iteration: int = -1
    iter_W: int = -1
    
    # Calculation settings
    excitation_type: str = "normal"
    n_spins: int = 1
    qpac_enabled: bool = False
    tda_enabled: bool = False
    calc_pi_on_fly: bool = False
    debug_enabled: bool = False
    
    # Performance
    n_jobs: int = -1
    monitoring_enabled: bool = False
    
    @classmethod
    def from_args(cls, args):
        """Create configuration from parsed arguments."""
        return cls(
            input_file=args.input,
            sim_file=args.sim,
            output_file=args.output,
            int_path=args.int_path,
            ir_file=args.ir_file,
            pi_file=args.pi_file,
            beta=args.beta,
            iteration=args.iter,
            iter_W=args.iter_W,
            excitation_type=args.type,
            n_spins=args.ns,
            qpac_enabled=bool(args.qpac),
            tda_enabled=bool(args.tda),
            calc_pi_on_fly=bool(args.calc_pi),
            debug_enabled=bool(args.debug),
            n_jobs=args.n_jobs,
            monitoring_enabled=args.monitor
        )


class SystemMonitor:
    """System monitoring utilities."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and MONITORING_AVAILABLE
        self.start_time = time.time()
    
    def print_system_info(self):
        """Print system information."""
        if not self.enabled:
            print("System monitoring not available")
            return
            
        try:
            cpu_physical = psutil.cpu_count(logical=False)
            cpu_logical = psutil.cpu_count(logical=True)
            memory = psutil.virtual_memory()
            
            print("=" * 70)
            print("SYSTEM INFORMATION")
            print("-" * 70)
            print(f"Physical CPU cores:    {cpu_physical}")
            print(f"Logical CPU cores:     {cpu_logical}")
            print(f"Total memory:          {memory.total / (1024**3):.2f} GB")
            print(f"Available memory:      {memory.available / (1024**3):.2f} GB")
            print(f"Used memory:           {memory.used / (1024**3):.2f} GB ({memory.percent:.1f}%)")
            print("=" * 70)
        except Exception as e:
            print(f"System monitoring not available: {e}")
    
    def print_process_info(self):
        """Print current process information."""
        if not self.enabled:
            return
            
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024**2)
            memory_percent = process.memory_percent()
            
            print("=" * 70)
            print("PROCESS INFORMATION")
            print("-" * 70)
            print(f"Process ID:            {process.pid}")
            print(f"Memory usage:          {memory_mb:.2f} MB ({memory_percent:.1f}%)")
            print(f"Number of threads:     {process.num_threads()}")
            print("=" * 70)
        except Exception as e:
            print(f"Process monitoring not available: {e}")
    
    def print_joblib_info(self, n_jobs):
        """Print joblib parallelization information."""
        if not self.enabled:
            print(f"Parallelization info: n_jobs={n_jobs}")
            return
            
        try:
            effective_jobs = effective_n_jobs(n_jobs)
            max_jobs = mp.cpu_count()
            
            print("=" * 70)
            print("PARALLELIZATION SETTINGS")
            print("-" * 70)
            print(f"Requested n_jobs:      {n_jobs}")
            print(f"Effective n_jobs:      {effective_jobs}")
            print(f"Maximum available:     {max_jobs}")
            print("n_jobs=-1 : use all available cores")
            print("n_jobs= 1 : serial execution")
            print("=" * 70)
        except Exception as e:
            print(f"Parallelization info: n_jobs={n_jobs} (monitoring unavailable): {e}")
    
    def estimate_memory_requirements(self, nao, nelec, niw):
        """Estimate memory requirements for BSE calculations."""
        occ = nelec // 2
        virt = nao - occ
        
        # Estimate major array sizes (complex128 = 16 bytes)
        VQ_mb = (nao**4 * 16) / (1024**2)  # VQ matrix
        Pi_mb = (niw * nao**4 * 16) / (1024**2)  # Pi matrix
        G2p_mb = (niw * 2 * occ * virt * 16) / (1024**2)  # G2p matrices
        working_mb = (occ**2 * virt**2 * 16 * 4) / (1024**2)  # Working arrays
        
        total_mb = VQ_mb + Pi_mb + G2p_mb + working_mb
        
        print("=" * 70)
        print("MEMORY ESTIMATES")
        print("-" * 70)
        print(f"Problem size: nao={nao}, nelec={nelec}, niw={niw}")
        print(f"VQ matrix:            {VQ_mb:.2f} MB")
        print(f"Pi matrix:            {Pi_mb:.2f} MB")
        print(f"G2p matrices:         {G2p_mb:.2f} MB")
        print(f"Working arrays:       {working_mb:.2f} MB")
        print(f"Total estimated:      {total_mb:.2f} MB ({total_mb/1024:.2f} GB)")
        print("=" * 70)
    
    def monitor_memory(self, stage: str = ""):
        """Simple memory monitoring."""
        if not self.enabled:
            return
            
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024**2)
            elapsed = time.time() - self.start_time
            print(f"{stage}: {memory_mb:.2f} MB | Time: {elapsed:.2f}s")
        except Exception:
            pass


class BSESolver:
    """BSE Casida equation solver."""
    
    def __init__(self, config: BSEConfig):
        self.config = config
        self.monitor = SystemMonitor(config.monitoring_enabled)
        
        # System properties (filled during calculation)
        self.nao = 0
        self.nelec = 0
        self.occ = 0
        self.virt = 0
        
        # Data storage
        self.valsMO = None
        self.vexMO = None
        self.results = {}
    
    def print_header(self):
        """Print calculation header."""
        print("=" * 70)
        print("BSE CASIDA EQUATION SOLVER")
        print("FOR MATUSBARA GREEN'S FUNCTION")
        print("Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.")
        print("                   Gaurav Harsha <gharsha@umich.edu>, University of Michigan.")

        print("-" * 70)
        print(f"Input file:        {self.config.input_file}")
        print(f"Integral file:     {self.config.int_path}")
        print(f"Simulation file:   {self.config.sim_file}")
        print(f"Output file:       {self.config.output_file}")
        print(f"Beta (inverse T):  {self.config.beta}")
        print(f"Excitation type:   {self.config.excitation_type}")
        print(f"Monitoring:        {'Enabled' if self.config.monitoring_enabled else 'Disabled'}")
        print("=" * 70)
        
        if self.config.monitoring_enabled:
            self.monitor.print_system_info()
            self.monitor.print_process_info()
            self.monitor.print_joblib_info(self.config.n_jobs)
            
        print("*" * 70)
        print(f"    Starting Casida solver for scGW iteration: {self.config.iteration}    ")
        print(f"    iter = -1 means using the last iteration in the sim file.    ")
        print("*" * 70)
        
    def load_input_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load input data from HDF5 files."""
        start = time.time()
        
        print("Reading IR file")
        with h5py.File(self.config.ir_file, "r") as f:
            wgrid = f["/bose/wsample"][()]
        wgrid = 2 * wgrid * np.pi / self.config.beta
        
        print("Reading input file")
        with h5py.File(self.config.input_file, 'r') as f:
            rSk = f["/HF/S-k"][()].view(complex)
            rSk = rSk.reshape(rSk.shape[:-1])
            rFk_input = f["/HF/Fock-k"][()].view(complex)
            self.nao = f["/params/nao"][()]
            self.nelec = f["/params/nel_cell"][()]
        
        self.occ = self.nelec // 2
        self.virt = self.nao - self.occ
        
        print("Reading sim file")
        with h5py.File(self.config.sim_file, 'r') as f:
            it = f["iter"][()] if self.config.iteration == -1 else self.config.iteration
            
            if it == 1:
                print("Reading the HF level Fock matrix for G0W0.")
                rFk = rFk_input
            else:
                rFk = f[f"iter{it}/Fock-k"][()].view(complex)
                
            rFk = rFk.reshape(rFk.shape[:-1])
            rSigmak = f[f"iter{it}/Selfenergy/data"][()].view(complex)
            rSigmak = rSigmak.reshape(rSigmak.shape[:-1])
            mu = f[f"iter{it}/mu"][()]
        
        end = time.time()
        print("*" * 70)
        print(f"    Reading files: {end - start:.4f} secs    ")
        print("*" * 70)
        
        # Store for later use
        self.results['rSk'] = rSk
        self.results['rFk'] = rFk
        self.results['rSigmak'] = rSigmak
        self.results['mu'] = mu
        self.results['wgrid'] = wgrid
        
        # Print memory estimates and current usage
        if self.config.monitoring_enabled:
            niw = len(wgrid)
            self.monitor.estimate_memory_requirements(self.nao, self.nelec, niw)
            self.monitor.monitor_memory("After file reading")
        
        return rFk, rSk
    
    def solve_molecular_orbitals(self, rFk: np.ndarray, rSk: np.ndarray):
        """Solve for molecular orbital eigenvalues and eigenvectors."""
        start = time.time()
        print("*" * 70)
        print("    Solving for MO eigenvalues    ")
        print("*" * 70)
        
        self.valsMO, self.vexMO = casida.solveMO(rFk, rSk)
        mo_coeff_gamma = np.zeros((self.config.n_spins, 1, self.nao, self.nao), dtype=np.complex128)
        mo_coeff_gamma[0, 0, :, :] = self.vexMO[...]
        mo_coeff = 1.0 * mo_coeff_gamma
        mo_coeff_adj = np.einsum('skpq -> skqp', mo_coeff.conj())
        
        # Pade interpolation using Sigma if enabled
        if self.config.qpac_enabled:
            print("Getting QP ac energy levels from self-energy.")
            Sigma_tk_int = np.einsum(
                'skab, tskbc, skcd -> tskad',
                mo_coeff_adj, self.results['rSigmak'], mo_coeff,
                optimize=True
            )
            valsMO_qp = qp.padeSigma(
                Sigma_tk_int, self.valsMO, self.config.beta, 
                self.results['mu'], self.config.ir_file
            )
            self.valsMO = valsMO_qp
        
        end = time.time()
        print("*" * 70)
        print(f"    Solve MO eigenvalues: {end - start:.4f} secs    ")
        print("*" * 70)
        
        return mo_coeff_adj, mo_coeff
    
    def prepare_interaction_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare VQ and Pi matrices."""
        start = time.time()
        
        # Load VQ matrix
        VQ_ao = ct.readVQ(self.config.int_path + "VQ_0.h5")
        nQ = VQ_ao.shape[1]
        VQ = casida.VQ_ao2mo(VQ_ao, self.vexMO[0, 0, :])
        
        # Handle Pi matrix (either calculate or read from file)
        if self.config.calc_pi_on_fly:
            print(f"Calculating Pi on the fly from iteration {self.config.iter_W} of sim file.")
            tildeP_tau = ct.getPtilde(
                self.config.iter_W, self.nao, nQ,
                tau_h5=self.config.ir_file,
                int_path=self.config.int_path,
                sim_h5=self.config.sim_file
            )
            tildeP_iw = ct.tau2omegaFT(tildeP_tau, beta=self.config.beta, tau_h5=self.config.ir_file)
            niw = tildeP_iw.shape[0]
            tildeP_iw = tildeP_iw.reshape(niw, 1, nQ, nQ, 1)
            del tildeP_tau
            
            if self.config.monitoring_enabled:
                self.monitor.monitor_memory("After Pi calculation")
        else:
            tildeP_iw = ct.readPtilde(self.config.pi_file)
            print("! Because you chose to read from Pi file, the iteration of W you specified might not be used. !")
            
            if self.config.monitoring_enabled:
                self.monitor.monitor_memory("After Pi file reading")
        
        niw = tildeP_iw.shape[0]
        
        if self.config.monitoring_enabled:
            print(f" Processing {niw} imaginary frequency points (half)")
            print(f" Problem dimensions: nao={self.nao}, nelec={self.nelec}, occ={self.occ}, virt={self.virt}")
        
        return VQ, tildeP_iw
    
    def solve_bse_equations(self, VQ: np.ndarray, tildeP_iw: np.ndarray):
        """Solve BSE equations."""
        niw = tildeP_iw.shape[0]
        
        if not self.config.tda_enabled:
            # Full BSE calculation
            diffEps_ov = casida.mo2ovStat(
                casida.diffEpsMat(self.valsMO[0, 0, :]), self.nelec
            ).reshape(self.occ * self.virt, self.occ * self.virt)
            
            effVals_static, _, _, _ = casida.solveHstatic(
                tildeP_iw[niw//2, 0, :, :, 0], VQ, diffEps_ov, self.nelec, 
                self.config.excitation_type, tda=0
            )
            
            _, effVex_static, o_idx_li, v_idx_li = casida.solveHstatic(
                tildeP_iw[0, 0, :, :, 0], VQ, diffEps_ov, self.nelec, 
                self.config.excitation_type, tda=0
            )
            
            H2p_inf = casida.HinfDiagApprox(
                effVex_static, VQ, self.valsMO, self.nelec, self.config.excitation_type
            )
            
            G2p_iw_init_inv = casida.initG2p_inv(H2p_inf, self.config.ir_file, self.config.beta)
            niw = G2p_iw_init_inv.shape[0]
            
            G2p_iw_init = np.zeros(G2p_iw_init_inv.shape, dtype=np.complex128)
            for iw in range(niw):
                G2p_iw_init[iw, :] = 1 / G2p_iw_init_inv[iw, :]
            
            if self.config.monitoring_enabled:
                self.monitor.monitor_memory("Before updateG2p_alt")
                print(f" Starting updateG2p_alt with n_jobs={self.config.n_jobs}")
            
            G2p_iw_updated = casida.updateG2p_alt(
                G2p_iw_init_inv, tildeP_iw, effVex_static, VQ, self.nelec, self.config.n_jobs
            )
                
            if self.config.monitoring_enabled:
                self.monitor.monitor_memory("After updateG2p_alt")
                
            G2p_tau_updated = ct.omega2tauFT(G2p_iw_updated, self.config.beta, self.config.ir_file)
            
            wpole_data, S_data, _, res_norm_data = plasPole.fit_G_update(
                G2p_iw_updated, self.config.ir_file, beta=self.config.beta
            )
            
            self.results.update({
                'G2p_iw_init': G2p_iw_init,
                'G2p_iw_updated': G2p_iw_updated,
                'G2p_tau_updated': G2p_tau_updated,
                'H2p_inf': H2p_inf,
                'effVals_static': effVals_static,
                'pole_fit': wpole_data,
                'S_fit': S_data,
                'residual_norm_fit': res_norm_data,
                'occ_index': o_idx_li,
                'virt_index': v_idx_li
            })
            
        else:
            print("TDA approximation is not fully implemented yet. Exiting.")
            raise NotImplementedError("TDA approximation not implemented")
    
    def save_results(self):
        """Save calculation results to HDF5 file."""
        start = time.time()
        with h5py.File(self.config.output_file, 'w') as f:
            f["/G2pInit/data"] = self.results['G2p_iw_init']
            f["/G2pUpdated/data"] = self.results['G2p_iw_updated']
            f["/G2pUpdated_tau/data"] = self.results['G2p_tau_updated']
            f["/InfExcVals/data"] = self.results['H2p_inf']
            f["/StatExcVals/data"] = self.results['effVals_static']
            f["/PoleFit/data"] = self.results['pole_fit']
            f["/SFit/data"] = self.results['S_fit']
            f["/ResNorm/data"] = self.results['residual_norm_fit']
            f["/MOvectors/data"] = self.vexMO
            f["/MOvals/data"] = self.valsMO
            f["/Indices/occ"] = self.results['occ_index']
            f["/Indices/virt"] = self.results['virt_index']
        
        end = time.time()
        print("*" * 70)
        print(f"    Write output: {end - start:.4f} secs    ")
        print("*" * 70)
    
    def print_results(self):
        """Print calculation results."""
        print("=" * 90)
        print("EXCITATION EIGENVALUES (eV)")
        print("Only printing the first 20 positive eigenvalues from each calculation.")
        print("-" * 90)
        
        # Get eigenvalues
        static_pos = self.results['effVals_static'][self.results['effVals_static'] > 0].real
        static_ev = static_pos * AU2EV
        inf_pos = self.results['H2p_inf'][self.results['H2p_inf'] > 0].real
        inf_ev = inf_pos * AU2EV
        dyn_pos = self.results['pole_fit'][self.results['S_fit'] > 0].real
        dyn_ev = dyn_pos * AU2EV
        
        occ_indices = self.results['occ_index'][self.results['H2p_inf'] > 0]
        virt_indices = self.results['virt_index'][self.results['H2p_inf'] > 0]
        
        # Determine how many to display (first 20)
        n_static = min(20, len(static_ev))
        n_inf = min(20, len(inf_ev))
        n_dyn = min(20, len(dyn_ev))
        n_occ_indices = min(20, len(occ_indices))
        n_virt_indices = min(20, len(virt_indices))
        n_display = max(n_static, n_inf, n_dyn, n_occ_indices, n_virt_indices)
        
        if n_display > 0:
            print(f"{'Index':>5} {'Static (eV)':>15} {'Infinite (eV)':>15} {'Dynamic (eV)':>15} {'Occupied MO':>15} {'Virtual MO':>15}")
            print("-" * 90)
            
            for i in range(n_display):
                static_val = static_ev[i] if i < len(static_ev) else None
                inf_val = inf_ev[i] if i < len(inf_ev) else None
                dyn_val = dyn_ev[i] if i < len(dyn_ev) else None
                occ_idx = occ_indices[i] if i < len(occ_indices) else None
                virt_idx = virt_indices[i] if i < len(virt_indices) else None
                
                static_str = f"{static_val:10.4f}" if static_val is not None else "    N/A    "
                inf_str = f"{inf_val:10.4f}" if inf_val is not None else "    N/A    "
                dyn_str = f"{dyn_val:10.4f}" if dyn_val is not None else "    N/A    "
                occ_str = f"{occ_idx:5d}" if occ_idx is not None else " N/A "
                virt_str = f"{virt_idx:5d}" if virt_idx is not None else " N/A "

                print(f"{i+1:5d} {static_str:>15} {inf_str:>15} {dyn_str:>15} {occ_str:>15} {virt_str:>15}")
                
            print("......")
        else:
            print("No positive eigenvalues found in either calculation")

        print("=" * 90)
    
    def run(self):
        """Main execution method."""
        self.print_header()
        
        # Load input data
        rFk, rSk = self.load_input_data()
        
        # Solve molecular orbitals
        mo_coeff_adj, mo_coeff = self.solve_molecular_orbitals(rFk, rSk)
        
        # Prepare interaction matrices
        VQ, tildeP_iw = self.prepare_interaction_matrices()
        
        # Solve BSE equations
        start_bse = time.time()
        self.solve_bse_equations(VQ, tildeP_iw)
        end_bse = time.time()
        print("*" * 70)
        print(f"    BSE Casida equation: {end_bse - start_bse:.4f} secs    ")
        print("*" * 70)
        
        # Print results
        self.print_results()
        
        # Save results
        self.save_results()
        
        # Final monitoring summary
        if self.config.monitoring_enabled:
            print("\n" + "=" * 70)
            print("FINAL SYSTEM STATUS")
            print("-" * 70)
            self.monitor.monitor_memory("Final memory usage")
            try:
                if MONITORING_AVAILABLE:
                    process = psutil.Process()
                    # peak_memory = process.memory_info().rss / (1024**2)
                    # print(f"Peak memory usage:     {peak_memory:.2f} MB")
                    print(f"Final thread count:    {process.num_threads()}")
                print("=" * 70)
            except Exception:
                print("Final monitoring unavailable")
                print("=" * 70)

def create_argument_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="BSE Casida equation solver with object-oriented design"
    )
    
    # File paths
    parser.add_argument("--input", type=str, default="input.h5",
                        help="Input of scGW")
    parser.add_argument("--sim", type=str, default="sim.h5",
                        help="Output of scGW")
    parser.add_argument("--output", type=str, default="bse.h5",
                        help="Output of BSE")
    parser.add_argument("--int_path", type=str, default="df_hf_int/",
                        help="Path for the VQ.h5 file")
    parser.add_argument("--ir_file", type=str, default=None,
                        help="HDF5 file that contains information about the IR grid.")
    parser.add_argument("--pi_file", type=str, default="p_iw_tilde_q0.h5",
                        help="HDF5 file that contains information about non-interacting susceptibility.")
    
    # Physical parameters
    parser.add_argument("--beta", type=float, default=1000, 
                        help="Inverse temperature")
    parser.add_argument("--iter", type=int, default=-1,
                        help="Iteration number of the scGW cycle to use for continuation")
    parser.add_argument("--iter_W", type=int, default=-1,
                        help="Iteration number of W you want to use. Default -1 means using the latest W.")
    
    # Calculation settings
    parser.add_argument("--type", type=str, default="normal",
                        help="Type of excitations.")
    parser.add_argument("--ns", type=int, default=1,
                        help="Number of spins")
    parser.add_argument("--qpac", type=int, default=0,
                        help="Enable QP AC?")
    parser.add_argument("--tda", type=int, default=0,
                        help="Enable TD approximation")
    parser.add_argument("--calc_pi", type=bool, default=False,
                        help="Option to calculate Pi on the fly. Not implemented yet.")
    parser.add_argument("--debug", type=int, default=0,
                        help="Enable debugging")
    
    # Performance
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="Number of parallel jobs for frequency loops (-1 = all cores, 1 = serial)")
    parser.add_argument("--monitor", action="store_true",
                        help="Enable system monitoring and memory tracking")
    
    return parser
