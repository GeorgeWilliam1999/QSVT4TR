from abc import ABC, abstractmethod
from utils.state_vector_machine.state_event_model.state_event_model import Event, Segment
from itertools import product, count
from scipy.sparse import eye
from scipy.sparse.linalg import cg
import numpy as np
import time
from scipy.linalg import block_diag
from numpy import array, cross
from numpy.linalg import solve, norm
import cProfile
import pstats
from qiskit.circuit import QuantumCircuit, QuantumRegister


class Hamiltonian(ABC):
    @abstractmethod
    def construct_hamiltonian(self, event: Event):
        pass
    
    @abstractmethod
    def evaluate(self, solution):
        pass


def count_gates_on_qubit(circuit, qubit):
  """
  Counts the number of gates applied to a specific qubit in a circuit.

  Args:
      circuit: The QuantumCircuit object.
      qubit: The Qubit object representing the qubit of interest.

  Returns:
      int: The number of gates applied to the specified qubit.
  """
  gate_count = 0
  for instr in circuit.data:
    if qubit in instr[1]:
      gate_count += 1
  return gate_count


def upscale_pow2(A,b):
    #add a constant the same as the original matrix for this number 
    m = A.shape[0]
    d = int(2**np.ceil(np.log2(m)) - m)
    if d > 0:
        A_tilde = np.block([[A, np.zeros((m, d),dtype=np.float64)],[np.zeros((d, m),dtype=np.float64), 3*np.eye(d,dtype=np.float64)]])
        b_tilde = np.block([b,b[:d]])
        return A_tilde, b_tilde
    else:
        return A, b

class SimpleHamiltonian(Hamiltonian):
    def __init__(self, epsilon, gamma, delta):
        self.epsilon                                    = epsilon
        self.gamma                                      = gamma
        self.delta                                      = delta
        self.Z                                          = None
        self.A                                          = None
        self.b                                          = None
        self.segments                                   = None
        self.segments_grouped                           = None
        self.n_segments                                 = None
    
    
    def construct_segments(self, event: Event):
        
        segments_grouped = []
        segments = []
        n_segments = 0
        segment_id = count()
        for idx in range(len(event.modules)-1):
            from_hits = event.modules[idx].hits
            to_hits = event.modules[idx+1].hits
            #print(to_hits)
            
            segments_group = []
            for from_hit, to_hit in product(from_hits, to_hits):
                seg = Segment(next(segment_id),from_hit, to_hit)
                segments_group.append(seg)
                segments.append(seg)
                n_segments = n_segments + 1
        
            segments_grouped.append(segments_group)
            
        
        self.segments_grouped = segments_grouped
        self.segments = segments
        self.n_segments = n_segments
        
    def construct_hamiltonian(self, event: Event):
        #pr = cProfile.Profile()
        #pr.enable()

        if self.segments_grouped is None:
            self.construct_segments(event)
        A = eye(self.n_segments,format='lil')*(-(self.delta+self.gamma))
        b = np.ones(self.n_segments)*self.delta
        for group_idx in range(len(self.segments_grouped) - 1):
            for seg_i, seg_j in product(self.segments_grouped[group_idx], self.segments_grouped[group_idx+1]):
                if seg_i.hit_to == seg_j.hit_from:
                    cosine = seg_i * seg_j
                    #print(cosine)
                    if abs(cosine - 1) < self.epsilon:
                        A[seg_i.segment_id, seg_j.segment_id] = A[seg_j.segment_id, seg_i.segment_id] =  1
        A = A.tocsc()
        
        self.A, self.b = -A, b
        #pr.disable()
        #stats = pstats.Stats(pr)
        #stats.sort_stats('cumulative')
        #stats.print_stats(10)
        return -A, b
    
        
    def construct_Z(self):
        def find_intersections(ham):
            def find_intersection(v1,v2):

                XA0 = v1
                XA1 = v2
                XB0 = array([0, 0, 0])
                XB1 = array([0, 0, 1])

                UA = (XA1 - XA0) / norm(XA1 - XA0)
                UB = (XB1 - XB0) / norm(XB1 - XB0)
                UC = cross(UB, UA); UC /= norm(UC)

                RHS = XB0 - XA0
                LHS = array([UA, -UB, UC]).T
                parameters = solve(LHS, RHS)
                intersection_point_B = XB0 + parameters[1] * UB
                return intersection_point_B
            intersections = []
            for track in range(len(ham.Z)):
                v1 = np.array([ham.Z[track].hit_from.x, ham.Z[track].hit_from.y, ham.Z[track].hit_from.z])
                v2 = np.array([ham.Z[track].hit_to.x, ham.Z[track].hit_to.y, ham.Z[track].hit_to.z])
                intercept = find_intersection(v1,v2)
                intersections.append(intercept[2])
            matrix = np.zeros((len(intersections), len(intersections)),float)
            np.fill_diagonal(matrix, intersections)
            return matrix
        
        self.Z = []
        for seg in self.segments_grouped:
            self.Z.append(seg)
        self.Z = self.Z[0] + self.Z[1]
        self.Z = find_intersections(self)
        
    
    def suzuki_trotter_circuit(self, A, time, num_slices=1):
        """Generate Suzuki-Trotter approximation for time evolution.

        Args:
            A (np.ndarray): The unitary matrix to be evolved.
            time (float): Total evolution time.
            num_slices (int): Number of Trotter slices.

        Returns:
            QuantumCircuit: The Suzuki-Trotter approximation circuit.
        """
        num_qubits = int(np.log2(A.shape[0]))
        qr = QuantumRegister(num_qubits, name="q")
        circuit = QuantumCircuit(qr, name="SuzukiTrotter")

        hamiltonian = -1j * A * time / num_slices

        for _ in range(num_slices):
            for i in range(num_qubits):
                theta, phi, lam = np.angle(hamiltonian[i, i]), -np.angle(hamiltonian[i, (i + 1) % num_qubits]), -np.angle(hamiltonian[i, (i - 1) % num_qubits])
                circuit.u(theta, phi, lam, qr[i])

        return circuit
    
    def solve_classicaly(self):
        if self.A is None:
            raise Exception("Not initialised")
        
        solution, _ = cg(self.A, self.b, atol=0)
        return solution
    

    def evaluate(self, solution):
        if self.A is None:
            raise Exception("Not initialised")
        
        if isinstance(solution, list):
            sol = np.array([solution, None])
        elif isinstance(solution, np.ndarray):
            if solution.ndim == 1:
                sol = solution[..., None]
            else: sol = solution
            
            
        return -0.5 * sol.T @ self.A @ sol + self.b.dot(sol)
        