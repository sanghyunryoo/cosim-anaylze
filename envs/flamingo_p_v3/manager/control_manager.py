import copy
import math
import random
import numpy as np

class ControlManager:
    def __init__(self, config):
        self.prev_action = None
        self.filtered_action = None
        self.prob = config["random"]["action_delay_prob"]
        # 원본 유지: 여기서는 아무 것도 더 안 건드림

    @staticmethod
    def pd_controller(kp, tq, q, kd, td, d):
        return kp * (tq - q) + kd * (td - d)

    def delay_filter(self, action):
        v = random.uniform(0, 1)
        delay_flag = True if self.prob > v else False
        if not delay_flag or self.prev_action is None:
            self.prev_action = action
            return action
        else:  # when delay_flag is True
            output = copy.deepcopy(self.prev_action)
            self.prev_action = action
            return output

    def reset(self):
        self.prev_action = None
        self.filtered_action = None
        # stabilize 테스트용 내부 상태도 함께 초기화 (lazy-init이므로 없어도 무방)
        if hasattr(self, "_v_hp"): delattr(self, "_v_hp")
        if hasattr(self, "_qdot_prev"): delattr(self, "_qdot_prev")

    # ------------------------------------------------------------------
    # 테스트용: self, action만 받아서 진동 억제(슬루+LPF+소프트클립+HP댐핑+패시비티)
    # q, qdot은 모의값(0)으로, dt/u_max/하이퍼파라미터는 내부 기본값 사용
    # ------------------------------------------------------------------
    # def delay_filter(self, action):
    #     """
    #     action : 정책 원출력 (list/np.ndarray, -1~1 가정)
    #     반환    : 안정화된 액션(동일 shape의 np.ndarray)
    #     """
    #     # ---- 기본 설정(테스트용) ----
    #     dt = 0.02                     # 50Hz 가정
    #     u_max = 1.0                   # 정규화 [-1,1]을 토크/명령으로 그대로 사용
    #     params = {
    #         "slew": 0.25,             # 스텝당 최대 변화량(정규화)
    #         "tau_lpf": dt,            # 1차 저역통과 시정수
    #         "kdh": 0.2,               # 고주파 속도 댐핑 이득
    #         "tau_hp": 2*dt,           # 하이패스 시정수
    #         "p_c1": 0.0, "p_c2": 0.5  # 패시비티 전력 한계
    #     }

    #     # 배열화 및 내부 상태 준비
    #     a_in = np.asarray(action, dtype=np.float64).copy()
    #     if self.prev_action is None:
    #         self.prev_action = np.zeros_like(a_in)
    #     if self.filtered_action is None:
    #         self.filtered_action = np.zeros_like(a_in)
    #     if not hasattr(self, "_v_hp"):
    #         self._v_hp = np.zeros_like(a_in)
    #     if not hasattr(self, "_qdot_prev"):
    #         self._qdot_prev = np.zeros_like(a_in)  # 테스트에선 0 속도 가정

    #     # 모의 상태(q, qdot): 테스트 간편화를 위해 0 사용
    #     q = np.zeros_like(a_in)
    #     qdot = np.zeros_like(a_in)
    #     u_max_vec = np.ones_like(a_in) * float(u_max)

    #     # 1) Slew-rate limiter
    #     da = a_in - self.prev_action
    #     da = np.clip(da, -params["slew"], +params["slew"])
    #     a_slew = self.prev_action + da
    #     self.prev_action = a_slew

    #     # 2) 1차 저역통과(LPF)
    #     alpha = dt / max(params["tau_lpf"], 1e-6)
    #     self.filtered_action = self.filtered_action + alpha * (a_slew - self.filtered_action)
    #     a_lpf = self.filtered_action

    #     # 3) 소프트 클립(tanh) + 스케일
    #     u = np.tanh(a_lpf) * u_max_vec

    #     # 4) 고주파 속도 댐핑(하이패스)
    #     beta = math.exp(-dt / max(params["tau_hp"], 1e-6))
    #     self._v_hp = beta * self._v_hp + beta * (qdot - self._qdot_prev)  # qdot=0 → 테스트에선 0 유지
    #     self._qdot_prev = qdot.copy()
    #     u = u - params["kdh"] * self._v_hp

    #     # 5) 패시비티 게이트(전력 제한) — 테스트에선 qdot=0이므로 보수적 무효화 수준
    #     P = float(np.dot(u, qdot))            # 0에 가까움
    #     P_max = params["p_c1"] + params["p_c2"] * float(np.linalg.norm(qdot))
    #     if P > P_max:
    #         u *= (P_max / (P + 1e-12))

    #     return u
