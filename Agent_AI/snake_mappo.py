# snake_mappo.py (개선판)
# - 판 5배 확대(기본 grid=55)
# - 먹이 먹으면 지렁이 길이 영구 증가
# - FPS 기본 4로 느리게
# 실행 예시:
#   pip install torch pygame
#   python snake_mappo.py --mode mappo --render
#   python snake_mappo.py --mode single --render --grid 55 --fps 3 --cell 14

import argparse
import random
import numpy as np
import pygame
import torch
import torch.nn as nn
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 1) 스네이크(2인) 환경 정의
# -----------------------------
# 액션: 0=UP,1=RIGHT,2=DOWN,3=LEFT,4=STAY
ACTIONS = [(0,-1),(1,0),(0,1),(-1,0),(0,0)]

class SnakeMultiEnv:
    """
    두 지렁이가 같은 그리드에서 먹이를 먹으며 충돌을 피함.
    - 그리드 크기: N x N
    - 전형적인 스네이크 규칙: 먹으면 길이가 1 증가(영구)
    - 보상:
        +1   : 먹이를 먹은 에이전트에게
        -1   : 벽/몸/상대와 충돌 -> 에피소드 종료
        -0.01: 스텝 코스트(양쪽 모두)
    - 종료: 충돌 또는 최대 스텝 T 도달
    관측:
      - 전역 상태: [headA(x,y), headB(x,y), food(x,y)] 정규화 (6차원)
      - 개별 관측 o_i: [self(x,y), other(x,y), food(x,y)] 정규화 (6차원)
    """
    def __init__(self, N=55, T=400, init_len=3, seed=None):
        self.N = N
        self.T = T
        self.init_len = init_len
        self.rng = random.Random(seed if seed is not None else random.randint(0, 999999))
        self.reset()

    def reset(self):
        self.t = 0
        # 시작 위치 (멀찍이 떨어뜨림)
        self.body_A = [(1, 1)]
        self.body_B = [(self.N-2, self.N-2)]
        self.dir_A = (1, 0)    # RIGHT
        self.dir_B = (-1, 0)   # LEFT

        # 초기 길이 세팅(몸 길이 = init_len)
        while len(self.body_A) < self.init_len:
            last = self.body_A[-1]
            self.body_A.append((max(0, last[0]-1), last[1]))
        while len(self.body_B) < self.init_len:
            last = self.body_B[-1]
            self.body_B.append((min(self.N-1, last[0]+1), last[1]))

        self.food = self._spawn_food()
        return self._global_state(), self._obs()

    def _spawn_food(self):
        occupied = set(self.body_A + self.body_B)
        # 기본 랜덤 스폰
        while True:
            p = (self.rng.randint(0, self.N-1), self.rng.randint(0, self.N-1))
            if p not in occupied:
                return p

    def step(self, aA, aB):
        self.t += 1
        rA = rB = -0.01  # step cost

        # 액션 -> 방향 반영(STAY 제외)
        dA = ACTIONS[aA]
        dB = ACTIONS[aB]
        if dA != (0, 0): self.dir_A = dA
        if dB != (0, 0): self.dir_B = dB

        headA = self.body_A[0]
        headB = self.body_B[0]
        new_head_A = (headA[0] + self.dir_A[0], headA[1] + self.dir_A[1])
        new_head_B = (headB[0] + self.dir_B[0], headB[1] + self.dir_B[1])

        def out_of_bounds(p):
            return not (0 <= p[0] < self.N and 0 <= p[1] < self.N)

        dead = False

        # --- 충돌 판정(전형적 스네이크 규칙) ---
        # 자기 몸/상대 몸(현재 프레임 기준)과의 충돌, 벽 충돌
        if out_of_bounds(new_head_A): dead = True
        if out_of_bounds(new_head_B): dead = True

        # 머리-머리 충돌
        if new_head_A == new_head_B:
            dead = True

        # 머리-몸 충돌(자기/상대)
        # tail 이동 타이밍까지 엄밀히 따지면 더 복잡하지만, 간단하게 현재 몸 기준으로 체크
        if (new_head_A in self.body_A[1:]) or (new_head_A in self.body_B):
            dead = True
        if (new_head_B in self.body_B[1:]) or (new_head_B in self.body_A):
            dead = True

        # --- 이동 및 먹이 처리 ---
        ate_A = (new_head_A == self.food)
        ate_B = (new_head_B == self.food)

        # 머리를 앞으로 붙임
        self.body_A.insert(0, new_head_A)
        self.body_B.insert(0, new_head_B)

        # 먹지 않았으면 꼬리 제거(길이 유지), 먹었으면 꼬리 유지(길이 +1)
        if not ate_A:
            self.body_A.pop()   # 꼬리 제거
        else:
            rA += 1.0
        if not ate_B:
            self.body_B.pop()
        else:
            rB += 1.0

        # 먹이가 먹혔다면 새 스폰
        if ate_A or ate_B:
            self.food = self._spawn_food()

        # 종료 처리
        done = dead or (self.t >= self.T)
        if dead:
            rA -= 1.0
            rB -= 1.0

        return (self._global_state(), self._obs(), (rA, rB), done)

    def _global_state(self):
        hAx, hAy = self.body_A[0]
        hBx, hBy = self.body_B[0]
        fx,  fy  = self.food
        s = np.array([
            hAx/(self.N-1), hAy/(self.N-1),
            hBx/(self.N-1), hBy/(self.N-1),
            fx/(self.N-1),  fy/(self.N-1)
        ], dtype=np.float32)
        return s

    def _obs(self):
        hAx, hAy = self.body_A[0]
        hBx, hBy = self.body_B[0]
        fx,  fy  = self.food
        oA = np.array([
            hAx/(self.N-1), hAy/(self.N-1),
            hBx/(self.N-1), hBy/(self.N-1),
            fx/(self.N-1),  fy/(self.N-1)
        ], dtype=np.float32)
        oB = np.array([
            hBx/(self.N-1), hBy/(self.N-1),
            hAx/(self.N-1), hAy/(self.N-1),
            fx/(self.N-1),  fy/(self.N-1)
        ], dtype=np.float32)
        return (oA, oB)

# -----------------------------
# 2) 네트워크와 PPO 구성
# -----------------------------
def mlp(in_dim, out_dim, hidden=128):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out_dim)
    )

class SinglePPO(nn.Module):
    def __init__(self, state_dim, action_dim_per_agent=5, num_agents=2, hidden=128):
        super().__init__()
        self.num_agents = num_agents
        self.action_dim = action_dim_per_agent
        self.policy = mlp(state_dim, action_dim_per_agent*num_agents, hidden)
        self.value  = mlp(state_dim, 1, hidden)

    def act(self, state):
        logits = self.policy(state)  # [B, 5*2]
        logits = logits.view(-1, self.num_agents, self.action_dim)
        dists  = [Categorical(logits=logits[:, i, :]) for i in range(self.num_agents)]
        acts   = torch.stack([d.sample() for d in dists], dim=1)              # [B,2]
        logp   = torch.stack([d.log_prob(acts[:, i]) for i,d in enumerate(dists)], dim=1).sum(dim=1)
        v      = self.value(state).squeeze(-1)
        return acts, logp, v

    def evaluate_actions(self, states, actions):
        logits = self.policy(states).view(-1, self.num_agents, self.action_dim)
        dists  = [Categorical(logits=logits[:, i, :]) for i in range(self.num_agents)]
        logp   = torch.stack([dists[i].log_prob(actions[:, i]) for i in range(self.num_agents)], dim=1).sum(dim=1)
        ent    = torch.stack([dists[i].entropy()             for i in range(self.num_agents)], dim=1).sum(dim=1)
        v      = self.value(states).squeeze(-1)
        return logp, ent, v

class MAPPOPolicy(nn.Module):
    def __init__(self, obs_dim_agent, state_dim_global, action_dim=5, hidden=128):
        super().__init__()
        self.pi_A  = mlp(obs_dim_agent, action_dim, hidden)
        self.pi_B  = mlp(obs_dim_agent, action_dim, hidden)
        self.critic= mlp(state_dim_global, 1, hidden)

    def act(self, obsA, obsB, state_global):
        la, lb = self.pi_A(obsA), self.pi_B(obsB)
        dA, dB = Categorical(logits=la), Categorical(logits=lb)
        aA, aB = dA.sample(), dB.sample()
        logp   = dA.log_prob(aA) + dB.log_prob(aB)
        v      = self.critic(state_global).squeeze(-1)
        return torch.stack([aA, aB], dim=1), logp, v

    def evaluate_actions(self, obsA, obsB, states, actions):
        la, lb = self.pi_A(obsA), self.pi_B(obsB)
        dA, dB = Categorical(logits=la), Categorical(logits=lb)
        logp   = dA.log_prob(actions[:,0]) + dB.log_prob(actions[:,1])
        ent    = dA.entropy() + dB.entropy()
        v      = self.critic(states).squeeze(-1)
        return logp, ent, v

# -----------------------------
# 3) PPO 유틸
# -----------------------------
class Rollout:
    def __init__(self):
        self.states=[]; self.obsA=[]; self.obsB=[]
        self.actions=[]; self.logp=[]; self.rews=[]
        self.dones=[]; self.values=[]

    def add(self, s,oA,oB,a,lp,r,d,v):
        self.states.append(s); self.obsA.append(oA); self.obsB.append(oB)
        self.actions.append(a); self.logp.append(lp)
        self.rews.append(r); self.dones.append(d); self.values.append(v)

    def to_tensors(self):
        arr = lambda x,dt=torch.float32: torch.tensor(np.array(x), dtype=dt, device=device)
        return (arr(self.states), arr(self.obsA), arr(self.obsB),
                torch.tensor(np.array(self.actions), dtype=torch.long, device=device),
                arr(self.logp), arr(self.rews), arr(self.dones), arr(self.values))

def gae_adv(rews, dones, vals, gamma=0.99, lam=0.95):
    T = len(rews)
    adv = torch.zeros(T, device=device)
    last = 0
    for t in reversed(range(T)):
        next_val = 0.0 if t==T-1 else vals[t+1]
        nonterm  = 1.0 - dones[t]
        delta    = rews[t] + gamma*next_val*nonterm - vals[t]
        last     = delta + gamma*lam*nonterm*last
        adv[t]   = last
    ret = adv + vals
    adv = (adv-adv.mean())/(adv.std()+1e-8)
    return adv, ret

def ppo_update(eval_fn, opt, states, obsA, obsB, actions, old_logp, returns, adv,
               clip=0.2, vf_coef=0.5, ent_coef=0.01, epochs=4, bs=256):
    N = states.size(0)
    idx = np.arange(N)
    for _ in range(epochs):
        np.random.shuffle(idx)
        for st in range(0, N, bs):
            mb = idx[st:st+bs]
            logp, ent, v = eval_fn(states[mb], obsA[mb], obsB[mb], actions[mb])
            ratio = torch.exp(logp - old_logp[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1-clip, 1+clip) * adv[mb]
            pol_loss = -torch.min(surr1, surr2).mean()
            v_loss   = 0.5*(returns[mb]-v).pow(2).mean()
            loss     = pol_loss + vf_coef*v_loss - ent_coef*ent.mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 0.5)
            opt.step()

# -----------------------------
# 4) 파이게임 렌더러
# -----------------------------
class SnakeRenderer:
    def __init__(self, N=55, cell=16, fps=4, title="Snake MAPPO Demo"):
        pygame.init()
        self.N=N; self.cell=cell; self.fps=fps
        self.scr = pygame.display.set_mode((N*cell, N*cell))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()

    def draw(self, env: SnakeMultiEnv):
        self.scr.fill((30,30,35))
        # 격자
        for i in range(self.N):
            pygame.draw.line(self.scr,(50,50,60),(0,i*self.cell),(self.N*self.cell,i*self.cell),1)
            pygame.draw.line(self.scr,(50,50,60),(i*self.cell,0),(i*self.cell,self.N*self.cell),1)
        # 먹이
        fx,fy = env.food
        pygame.draw.rect(self.scr,(220,70,70), (fx*self.cell, fy*self.cell, self.cell, self.cell))
        # A 몸통
        for i,p in enumerate(env.body_A):
            c = (80,200,120) if i>0 else (40,240,140)
            pygame.draw.rect(self.scr,c, (p[0]*self.cell, p[1]*self.cell, self.cell, self.cell))
        # B 몸통
        for i,p in enumerate(env.body_B):
            c = (120,150,240) if i>0 else (90,120,255)
            pygame.draw.rect(self.scr,c, (p[0]*self.cell, p[1]*self.cell, self.cell, self.cell))
        pygame.display.flip()
        self.clock.tick(self.fps)

    def maybe_quit(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); raise SystemExit

# -----------------------------
# 5) 학습/실행 루프
# -----------------------------
def train_and_render(mode="mappo", iters=200, horizon=256, grid=55, fps=4, cell=16, render=True):
    env = SnakeMultiEnv(N=grid, T=600, init_len=3, seed=random.randint(0,999999))
    state_dim=6   # [Ax,Ay,Bx,By,Fx,Fy]
    obs_dim=6     # [self(x,y), other(x,y), food(x,y)]
    action_dim=5

    if mode=="single":
        model = SinglePPO(state_dim, action_dim_per_agent=action_dim, num_agents=2).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        def act_fn(s,oA,oB):
            with torch.no_grad():
                S = torch.tensor(s,dtype=torch.float32,device=device).unsqueeze(0)
                acts, lp, v = model.act(S)
            return acts.squeeze(0).cpu().numpy(), lp.item(), v.item()
        def eval_fn(S,OA,OB,AC):
            return model.evaluate_actions(S, AC)
    else:
        model = MAPPOPolicy(obs_dim_agent=obs_dim, state_dim_global=state_dim, action_dim=action_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        def act_fn(s,oA,oB):
            with torch.no_grad():
                S  = torch.tensor(s,dtype=torch.float32,device=device).unsqueeze(0)
                OA = torch.tensor(oA,dtype=torch.float32,device=device).unsqueeze(0)
                OB = torch.tensor(oB,dtype=torch.float32,device=device).unsqueeze(0)
                acts, lp, v = model.act(OA,OB,S)
            return acts.squeeze(0).cpu().numpy(), lp.item(), v.item()
        def eval_fn(S,OA,OB,AC):
            return model.evaluate_actions(OA,OB,S,AC)

    renderer = SnakeRenderer(N=grid, cell=cell, fps=fps, title=f"Snake {mode.upper()}")

    ep_returns=[]; recent_len=20
    state,(oA,oB) = env.reset()

    for it in range(1, iters+1):
        buf = Rollout()
        steps=0; ep_sum=0.0
        while steps < horizon:
            if render:
                renderer.maybe_quit()
                renderer.draw(env)
            acts, lp, v = act_fn(state,oA,oB)
            aA, aB = int(acts[0]), int(acts[1])
            next_state,(noA,noB),(rA,rB),done = env.step(aA,aB)
            r = rA + rB  # 팀 보상(필요 시 개별로 바꿔도 무방)
            buf.add(state,oA,oB,np.array([aA,aB]), lp, r, float(done), v)
            ep_sum += r
            steps += 1
            state,oA,oB = next_state,noA,noB
            if done:
                ep_returns.append(ep_sum)
                state,(oA,oB) = env.reset(); ep_sum=0.0

        S,OA,OB,AC,OLP,REW,DON,VAL = buf.to_tensors()
        ADV,RET = gae_adv(REW,DON,VAL, gamma=0.99, lam=0.95)

        def _eval(s,oa,ob,ac):
            return eval_fn(s,oa,ob,ac)
        ppo_update(_eval, opt, S,OA,OB,AC, OLP,RET,ADV, clip=0.2, vf_coef=0.5, ent_coef=0.01, epochs=4, bs=256)

        if it % 10 == 0:
            avg = float(np.mean(ep_returns[-recent_len:])) if ep_returns else 0.0
            print(f"[{mode.upper()}] iter {it:4d} | recent avg team return: {avg:.3f}")

    # 마지막 시연
    if render:
        print("시연 시작(창 닫으면 종료).")
        state,(oA,oB) = env.reset()
        while True:
            renderer.maybe_quit()
            renderer.draw(env)
            acts,_,_ = act_fn(state,oA,oB)
            aA,aB = int(acts[0]), int(acts[1])
            state,(oA,oB),_,done = env.step(aA,aB)
            if done:
                state,(oA,oB) = env.reset()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single","mappo"], default="mappo")
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--horizon", type=int, default=256)
    ap.add_argument("--grid", type=int, default=55, help="그리드 크기(NxN)")
    ap.add_argument("--fps", type=int, default=4, help="렌더 FPS (낮을수록 느림)")
    ap.add_argument("--cell", type=int, default=16, help="셀 픽셀 크기")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()
    train_and_render(mode=args.mode, iters=args.iters, horizon=args.horizon,
                     grid=args.grid, fps=args.fps, cell=args.cell, render=args.render)

if __name__ == "__main__":
    main()
