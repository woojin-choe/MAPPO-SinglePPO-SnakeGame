# snake_centralized.py
# 중앙집중 제어 + 흡수 규칙(머리가 다른 지렁이의 몸에 닿으면 몸 쪽이 상대 전체 길이 흡수)

import argparse, random
from collections import deque
import numpy as np
import pygame

# 액션: UP, RIGHT, DOWN, LEFT, STAY
ACTIONS = [(0,-1),(1,0),(0,1),(-1,0),(0,0)]
UP, RIGHT, DOWN, LEFT, STAY = range(5)

# -------------------------
# 환경
# -------------------------
class MultiSnakeEnv:
    def __init__(self, N=35, num_snakes=2, foods=50, init_len=3, T=2000, seed=None):
        self.N = N
        self.num_snakes = num_snakes
        self.food_target = foods
        self.init_len = init_len
        self.T = T
        self.rng = random.Random(seed if seed is not None else random.randint(0, 10**9))
        self.reset()

    def reset(self):
        self.t = 0
        # 지렁이 초기 배치(멀리 떨어뜨림)
        self.snakes = []
        self.alive = [True]*self.num_snakes
        starts = [
            (1,1),(self.N-2,self.N-2),(1,self.N-2),(self.N-2,1)
        ]
        dirs = [(1,0),(-1,0),(1,0),(-1,0)]
        for i in range(self.num_snakes):
            head = starts[i % len(starts)]
            direction = dirs[i % len(dirs)]
            body = deque([head])
            # 초기 길이 확보
            while len(body) < self.init_len:
                last = body[-1]
                prev = (max(0, min(self.N-1, last[0]-direction[0])),
                        max(0, min(self.N-1, last[1]-direction[1])))
                if prev == last:
                    break
                body.append(prev)
            self.snakes.append({
                "body": body,           # deque of (x,y)
                "dir": direction,       # (dx,dy)
                "grow": 0,              # 성장 대기칸(양수면 그만큼 꼬리를 안 줄임)
                "color_head": (40,240,140) if i%2==0 else (90,120,255),
                "color_body": (80,200,120) if i%2==0 else (120,150,240)
            })

        # 먹이(고정): 중복/충돌 피해서 여러 개 생성
        self.food = set()
        occupied = self._occupied_set()
        while len(self.food) < self.food_target:
            p = (self.rng.randint(0,self.N-1), self.rng.randint(0,self.N-1))
            if p not in occupied:
                self.food.add(p)

        return self._get_local_obs_all()

    def _occupied_set(self):
        occ = set()
        for s in self.snakes:
            for p in s["body"]:
                occ.add(p)
        return occ

    def _in_bounds(self, p):
        return 0 <= p[0] < self.N and 0 <= p[1] < self.N

    def _local_obs(self, head):
        # 3x3 패치(헤드 중심): 벽/먹이 표시
        # 채널 2개: [벽(1/0), 먹이(1/0)]
        patch = np.zeros((3,3,2), dtype=np.float32)
        hx, hy = head
        for dy in [-1,0,1]:
            for dx in [-1,0,1]:
                x, y = hx+dx, hy+dy
                ix, iy = dx+1, dy+1
                if not self._in_bounds((x,y)):
                    patch[iy,ix,0] = 1.0  # 벽
                if (x,y) in self.food:
                    patch[iy,ix,1] = 1.0  # 먹이
        return patch.reshape(-1)  # 길이 18

    def _get_local_obs_all(self):
        obs = []
        for i,s in enumerate(self.snakes):
            head = s["body"][0]
            obs.append(self._local_obs(head))
        return obs  # 리스트 길이=num_snakes, 각 원소 shape=(18,)

    def step(self, actions):
        # actions: 길이 num_snakes, 각 액션 인덱스(0~4)
        self.t += 1

        # 1) 방향 갱신(STAY는 유지)
        for i, act in enumerate(actions):
            if not self.alive[i]: continue
            d = ACTIONS[act]
            if d != (0,0):
                self.snakes[i]["dir"] = d

        # 2) 이동 결과(새 머리 위치) 미리 계산
        new_heads = []
        pre_len = [len(s["body"]) for s in self.snakes]  # 흡수 길이 계산에 사용
        for i,s in enumerate(self.snakes):
            if not self.alive[i]:
                new_heads.append(None)
                continue
            hx, hy = s["body"][0]
            dx, dy = s["dir"]
            new_heads.append((hx+dx, hy+dy))

        # 3) 충돌 판정용 몸통 점유(이동 전 기준, 머리 제외)
        body_occ_by_snake = []
        all_body_occ = set()
        for s in self.snakes:
            body_ex_head = set(list(s["body"])[1:])
            body_occ_by_snake.append(body_ex_head)
            all_body_occ |= body_ex_head

        # 4) 사망/흡수 이벤트 판정
        dead = [False]*self.num_snakes
        capture_events = []  # (attacker_i, victim_j)
        # (a) 벽 충돌, (b) 자기/상대 몸통 충돌(흡수 후보), (c) 머리-머리 충돌
        for i,nh in enumerate(new_heads):
            if not self.alive[i]: 
                continue
            # 벽
            if nh is None or not self._in_bounds(nh):
                dead[i] = True
                continue
            # 머리-머리 충돌은 나중에 일괄 처리
        # 머리-머리
        for i in range(self.num_snakes):
            if not self.alive[i] or dead[i]: continue
            for j in range(i+1,self.num_snakes):
                if not self.alive[j] or dead[j]: continue
                if new_heads[i] == new_heads[j]:
                    dead[i] = True; dead[j] = True

        # 몸통 충돌(흡수 룰): 자기 몸은 제외하고, 다른 지렁이 몸에 닿은 경우만 흡수
        for i,nh in enumerate(new_heads):
            if not self.alive[i] or dead[i]: 
                continue
            # 이미 머리-머리로 죽었으면 처리하지 않음
            hit_any = False
            for j in range(self.num_snakes):
                if i == j: 
                    continue
                if nh in body_occ_by_snake[j]:
                    # i의 머리가 j의 몸통에 닿았다 → j가 i를 흡수
                    capture_events.append((i, j))
                    hit_any = True
                    break
            # 벽/몸 충돌로 죽음 플래그는 아직 두지 않음(흡수 우선 규칙 적용을 위해)
            # 흡수 이벤트가 있으면 i는 이동하지 않고 제거될 예정

        # 5) 흡수 처리: 피해자(victim)가 가해자(attacker)의 전체 길이 흡수
        #    동시에 여러 건 발생할 수 있으므로 피해자별 총 흡수량을 합산
        absorb_add = [0]*self.num_snakes
        attackers_to_kill = set()
        for atk, vic in capture_events:
            if not self.alive[atk] or not self.alive[vic]:
                continue
            if dead[atk]:  # 머리-머리 등으로 이미 죽었으면 흡수 안 함
                continue
            gain = pre_len[atk]
            absorb_add[vic] += gain
            attackers_to_kill.add(atk)

        # 6) 실제 이동/먹이/성장/사망 적용
        #    규칙:
        #     - 흡수된 공격자(attacker)는 즉시 제거(이동하지 않음)
        #     - 피해자(victim)는 grow에 길이 추가
        #     - 나머지 일반 이동 후, 먹이 먹으면 grow+1
        for i,s in enumerate(self.snakes):
            if not self.alive[i]:
                continue

            # 공격자로 지정되었거나 벽/머리-머리로 죽음이면 제거
            if i in attackers_to_kill or dead[i]:
                self.alive[i] = False
                continue

            nh = new_heads[i]
            # 정상 이동
            s["body"].appendleft(nh)

            # 먹이 처리
            ate = False
            if nh in self.food:
                ate = True
                self.food.remove(nh)
            if ate:
                s["grow"] += 1

            # 흡수로 인한 성장 추가
            if absorb_add[i] > 0:
                s["grow"] += absorb_add[i]

            # 성장량이 있으면 꼬리 유지(=길이 증가), 없으면 꼬리 제거
            if s["grow"] > 0:
                s["grow"] -= 1
            else:
                s["body"].pop()

        # 7) 종료 판단
        done = self.t >= self.T or all(a==False for a in self.alive)

        # 8) 먹이가 너무 줄었으면 다시 보충(옵션)
        if len(self.food) < self.food_target//2:
            occupied = self._occupied_set()
            while len(self.food) < self.food_target:
                p = (self.rng.randint(0,self.N-1), self.rng.randint(0,self.N-1))
                if p not in occupied:
                    self.food.add(p)

        obs = self._get_local_obs_all()
        return obs, done

# -------------------------
# 중앙 컨트롤러(규칙 기반)
# -------------------------
class CentralController:
    """
    하나의 중앙이 여러 지렁이를 동시에 제어.
    입력: 지렁이별 로컬관측(3x3×2=18 float)
    출력: 지렁이별 액션(UP/RIGHT/DOWN/LEFT/ STAY)
    정책: 단순 규칙 기반
      - 주변 3x3 중 먹이가 보이면 그 방향으로 이동
      - 없으면 머리 방향 유지, 안되면 안전한 임의 방향
    """
    def __init__(self, snake_indices):
        self.ids = snake_indices  # 이 중앙이 담당하는 지렁이 인덱스 리스트

    def decide(self, env: MultiSnakeEnv, obs_list):
        actions = {}
        # 먹이 있는 방향(3x3) 탐색
        def greedy_from_patch(patch18):
            patch = patch18.reshape(3,3,2)
            # 중앙(1,1). 먹이 채널=1
            dirs = [(0,-1),(1,0),(0,1),(-1,0)]
            idxs = [(1,0),(2,1),(1,2),(0,1)]  # 위/오/아래/왼
            for d,(ix,iy) in zip([UP,RIGHT,DOWN,LEFT], idxs):
                if patch[iy,ix,1] > 0.5 and patch[iy,ix,0] < 0.5:
                    return d
            return None

        def safe_fallback(head, current_dir):
            dx,dy = current_dir
            cand = [ (UP,(0,-1)), (RIGHT,(1,0)), (DOWN,(0,1)), (LEFT,(-1,0)) ]
            # 현재 방향 우선
            ordered = []
            for name,vec in cand:
                if vec == current_dir:
                    ordered = [(name,vec)] + [x for x in cand if x!=(name,vec)]
                    break
            if not ordered: ordered = cand
            body_occ = env._occupied_set()
            for name,vec in ordered:
                nx,ny = head[0]+vec[0], head[1]+vec[1]
                if 0 <= nx < env.N and 0 <= ny < env.N and (nx,ny) not in body_occ:
                    return name
            return STAY

        for sid in self.ids:
            if not env.alive[sid]:
                actions[sid] = STAY
                continue
            s = env.snakes[sid]
            head = s["body"][0]
            patch18 = obs_list[sid]
            g = greedy_from_patch(patch18)
            if g is not None:
                actions[sid] = g
            else:
                # 먹이가 안 보이면 현재 진행방향 유지 우선
                dx,dy = s["dir"]
                prefer = None
                if   (dx,dy)==(0,-1): prefer = UP
                elif (dx,dy)==(1,0):  prefer = RIGHT
                elif (dx,dy)==(0,1):  prefer = DOWN
                elif (dx,dy)==(-1,0): prefer = LEFT
                if prefer is not None:
                    nx,ny = head[0]+ACTIONS[prefer][0], head[1]+ACTIONS[prefer][1]
                    if 0 <= nx < env.N and 0 <= ny < env.N and (nx,ny) not in env._occupied_set():
                        actions[sid] = prefer
                    else:
                        actions[sid] = safe_fallback(head, s["dir"])
                else:
                    actions[sid] = safe_fallback(head, s["dir"])
        return actions

# -------------------------
# 렌더러
# -------------------------
class Renderer:
    def __init__(self, N, cell=18, fps=4, title="Centralized Snakes"):
        pygame.init()
        self.N=N; self.cell=cell; self.fps=fps
        self.scr = pygame.display.set_mode((N*cell, N*cell))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()

    def draw(self, env: MultiSnakeEnv):
        self.scr.fill((30,30,35))
        # 격자
        for i in range(self.N):
            pygame.draw.line(self.scr,(55,55,66),(0,i*self.cell),(self.N*self.cell,i*self.cell),1)
            pygame.draw.line(self.scr,(55,55,66),(i*self.cell,0),(i*self.cell,self.N*self.cell),1)
        # 먹이(고정)
        for fx,fy in env.food:
            pygame.draw.rect(self.scr,(220,70,70),(fx*self.cell, fy*self.cell, self.cell, self.cell))
        # 지렁이
        for i,s in enumerate(env.snakes):
            for k,p in enumerate(s["body"]):
                color = s["color_head"] if k==0 else s["color_body"]
                pygame.draw.rect(self.scr,color,(p[0]*self.cell, p[1]*self.cell, self.cell, self.cell))
        pygame.display.flip()
        self.clock.tick(self.fps)

    def maybe_quit(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); raise SystemExit

# -------------------------
# 실행 루프
# -------------------------
def run(mode="single", grid=35, foods=50, fps=4, cell=18, render=True, T=2000):
    if mode=="single":
        env = MultiSnakeEnv(N=grid, num_snakes=2, foods=foods, init_len=3, T=T)
        # 중앙 1개(두 마리 담당: 0,1)
        controllers = [CentralController([0,1])]
    else:
        env = MultiSnakeEnv(N=grid, num_snakes=4, foods=foods, init_len=3, T=T)
        # 중앙 2개(각각 두 마리씩)
        controllers = [CentralController([0,1]), CentralController([2,3])]

    renderer = Renderer(N=grid, cell=cell, fps=fps, title=f"Centralized Snakes - {mode.upper()}")
    obs_list = env.reset()

    while True:
        if render:
            renderer.maybe_quit()
            renderer.draw(env)

        # 각 중앙이 자신이 맡은 지렁이 관측만 사용해 액션 결정
        combined_actions = [STAY]*env.num_snakes
        for ctrl in controllers:
            actions = ctrl.decide(env, obs_list)
            for sid, act in actions.items():
                combined_actions[sid] = act

        obs_list, done = env.step(combined_actions)
        if done:
            obs_list = env.reset()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single","multi"], default="single")
    ap.add_argument("--grid", type=int, default=35, help="보드 크기(NxN)")
    ap.add_argument("--food", type=int, default=50, help="고정 먹이 개수")
    ap.add_argument("--fps", type=int, default=4, help="렌더 속도(FPS). 낮을수록 느림")
    ap.add_argument("--cell", type=int, default=18, help="셀 픽셀 크기")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()
    run(mode=args.mode, grid=args.grid, foods=args.food, fps=args.fps, cell=args.cell, render=args.render)

if __name__ == "__main__":
    main()
