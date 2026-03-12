import numpy as np

from pydrake.all import RollPitchYaw, Quaternion, RotationMatrix

from qsim.simulator import QuasistaticSimulator
from qsim_cpp import QuasistaticSimulatorCpp

from irs_rrt.rrt_params import IrsRrtProjectionParams
from irs_rrt.irs_rrt_3d import IrsRrt3D
from irs_rrt.irs_rrt_projection import IrsRrtProjection
from irs_rrt.contact_sampler import ContactSampler


class IrsRrtProjection3D(IrsRrtProjection):
    def __init__(
        self,
        rrt_params: IrsRrtProjectionParams,
        contact_sampler: ContactSampler,
        q_sim: QuasistaticSimulatorCpp,
        q_sim_py: QuasistaticSimulator,
        get_obj_pose_from_t,
        q_sim_smooth: QuasistaticSimulatorCpp | None = None,
        fixed_eef_1 = None,
        fixed_eef_2 = None
    ):
        super().__init__(rrt_params, contact_sampler, q_sim, q_sim_py, q_sim_smooth)
        self.get_obj_pose_from_t = get_obj_pose_from_t
        self.fixed_eef_1 = fixed_eef_1
        self.fixed_eef_2 = fixed_eef_2
    
    def sample_subgoal(self, t=None):
        """
        Sample a subgoal from the configuration space.
        """
        # sample robots
        subgoal = np.random.rand(self.dim_x)
        subgoal = self.q_lb + (self.q_ub - self.q_lb) * subgoal

        # sample obj pose
        if t is None:
            t0 = self.trajectory_ts[self.ts_i]
            t1 = self.trajectory_ts[self.ts_i + 1]
            t = np.random.rand() * (t1 - t0) + t0
            print(f"Sampling in [{t0};{t1}]: {t}")
        obj_pose_t = self.get_obj_pose_from_t(t)
        subgoal[self.q_u_indices_into_x] = obj_pose_t

        obj_rot_mat = Quaternion(obj_pose_t[:4]).rotation()
        obj_trans =  obj_pose_t[4:]

        if self.fixed_eef_1 is not None:
            subgoal[self.q_a_indices_into_x[:3]] = obj_rot_mat @ self.fixed_eef_1 + obj_trans
        
        if self.fixed_eef_2 is not None:
            subgoal[self.q_a_indices_into_x[3:]] = obj_rot_mat @ self.fixed_eef_2 + obj_trans        

        return subgoal