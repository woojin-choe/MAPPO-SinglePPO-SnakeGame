# snake_centralized.py
# - 빨간 먹이 여러 개를 무작위로 "고정" 배치
# - 각 지렁이는 헤드 기준 3x3(주변 1칸) 로컬 관측을 중앙에 전달
# - 중앙 컨트롤러가 각 지렁이 액션을 결정
# - 먹이를 먹으면 몸 길이 +1
# - single: 2마리 + 중앙1개 / multi: 4마리 + 중앙2개(각각 2마리 담당)
# 실행 예: python snake_centralized.py --mode multi --render

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
                "grow": 0,              # 성장 대기칸
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
        # 평탄화하여 반환(길이 18)
        return patch.reshape(-1)

    def _get_local_obs_all(self):
        # 각 지렁이 헤드의 로컬 관측
        obs = []
        for i,s in enumerate(self.snakes):
            head = s["body"][0]
            obs.append(self._local_obs(head))
        return obs  # 리스트 길이=num_snakes, 각 원소 shape=(18,)

    def step(self, actions):
        # actions: 길이 num_snakes, 각 액션 인덱스(0~4)
        self.t += 1

        # 먼저 방향 갱신(STAY는 유지)
        for i, act in enumerate(actions):
            if not self.alive[i]: continue
            d = ACTIONS[act]
            if d != (0,0):
                self.snakes[i]["dir"] = d

        # 이동 결과 미리 계산
        new_heads = []
        for i,s in enumerate(self.snakes):
            if not self.alive[i]:
                new_heads.append(None)
                continue
            hx, hy = s["body"][0]
            dx, dy = s["dir"]
            new_heads.append((hx+dx, hy+dy))

        # 충돌 판정(벽/머리-머리)
        # 몸통 점유(이동 전 상태 기준)
        body_occ = set()
        for s in self.snakes:
            for p in list(s["body"])[1:]:  # 머리 제외
                body_occ.add(p)

        dead = [False]*self.num_snakes
        # 벽/몸통/상호 머리충돌
        for i,nh in enumerate(new_heads):
            if not self.alive[i]: continue
            if nh is None or not self._in_bounds(nh):
                dead[i] = True
                continue
            if nh in body_occ:
                dead[i] = True
        # 머리끼리 같은 칸으로 이동하는 경우
        for i in range(self.num_snakes):
            if not self.alive[i]: continue
            for j in range(i+1,self.num_snakes):
                if not self.alive[j]: continue
                if new_heads[i] == new_heads[j]:
                    dead[i] = True; dead[j] = True

        # 실제 이동/성장/먹이 처리
        for i,s in enumerate(self.snakes):
            if not self.alive[i]: continue
            if dead[i]:
                self.alive[i] = False
                continue
            nh = new_heads[i]
            s["body"].appendleft(nh)
            ate = False
            if nh in self.food:
                ate = True
                self.food.remove(nh)
            if ate:
                s["grow"] += 1  # 길이 +1
            if s["grow"] > 0:
                s["grow"] -= 1  # 꼬리 유지(=성장)
            else:
                s["body"].pop() # 성장 없으면 꼬리 제거

        done = self.t >= self.T or all(a==False for a in self.alive)
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
        # 도우미: 먹이 있는 방향 찾기(3x3에서)
        def greedy_from_patch(patch18):
            patch = patch18.reshape(3,3,2)
            # 중앙 기준 인덱스(1,1). 먹이 채널=1
            # 우선순위: 상/우/하/좌
            dirs = [(0,-1),(1,0),(0,1),(-1,0)]
            idxs = [(1,0),(2,1),(1,2),(0,1)]  # 3x3 인덱스
            for d,(ix,iy) in zip([UP,RIGHT,DOWN,LEFT], idxs):
                if patch[iy,ix,1] > 0.5 and patch[iy,ix,0] < 0.5:  # 먹이 있고 벽 아님
                    return d
            return None

        # 안전한 대체 방향
        def safe_fallback(head, current_dir):
            # 현재 방향이 안전하면 유지
            dx,dy = current_dir
            cand = [ (UP,(0,-1)), (RIGHT,(1,0)), (DOWN,(0,1)), (LEFT,(-1,0)) ]
            # 현재 방향을 우선
            ordered = []
            for name,vec in cand:
                if vec == current_dir:
                    ordered = [(name,vec)] + [x for x in cand if x!=(name,vec)]
                    break
            if not ordered: ordered = cand
            for name,vec in ordered:
                nx,ny = head[0]+vec[0], head[1]+vec[1]
                if 0 <= nx < env.N and 0 <= ny < env.N:
                    # 다음 칸이 다른 몸통이면 피함(대충)
                    body_occ = env._occupied_set()
                    if (nx,ny) not in body_occ:
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
                # 유지가 불가하면 대체 방향
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

