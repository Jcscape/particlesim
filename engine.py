from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import List
import numpy as np
from .fields import Fields, CELL

@dataclass
class Params:
    n: int = 2500
    types: int = 6
    radius: float = 22.0
    fric: float = 0.968
    dt: float = 0.85
    bias: float = 0.15
    frange: float = 1.2
    fmul: float = 1.15
    food: float = 0.5
    reg: float = 0.012
    birth: float = 140.0
    maxage: int = 7000

@dataclass
class Agent:
    x: float; y: float
    vx: float; vy: float
    t: int
    age: int
    e: float
    hue: float
    sp: float
    se: float

class Engine:
    def __init__(self, width: int, height: int, prm: Params, seed: int=42):
        self.width = width; self.height = height
        self.prm = prm
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.fields = Fields(width, height, self.np_rng, prm.food)
        self.R = np.array([prm.radius*(0.8+0.5*self.np_rng.rand()) for _ in range(prm.types)], dtype=np.float32)
        self.F = self._regen_rules()
        self.A: List[Agent] = []
        self._build_spatial()
        self.reset_agents()

    def _build_spatial(self):
        self.gcols = math.ceil(self.width / CELL)
        self.grows = math.ceil(self.height / CELL)
        self.buckets = [-1]*(self.gcols*self.grows)
        self.nexts = []

    def _clear_spatial(self):
        for i in range(len(self.buckets)):
            self.buckets[i] = -1
        self.nexts = [-1]*len(self.A)

    def _insert_spatial(self, idx: int, x: float, y: float):
        cx = max(0, min(self.gcols-1, int(x//CELL)))
        cy = max(0, min(self.grows-1, int(y//CELL)))
        b = cy*self.gcols + cx
        self.nexts[idx] = self.buckets[b]
        self.buckets[b] = idx

    def _neighbors(self, x: float, y: float, rad: float, out: list):
        out.clear()
        r = int((rad+CELL)//CELL)
        cx = int(x//CELL); cy = int(y//CELL)
        for dy in range(-r, r+1):
            yy = cy+dy
            if yy<0 or yy>=self.grows: continue
            for dx in range(-r, r+1):
                xx = cx+dx
                if xx<0 or xx>=self.gcols: continue
                b = yy*self.gcols + xx
                j = self.buckets[b]
                while j!=-1:
                    out.append(j)
                    j = self.nexts[j]
        return out

    def _genome(self):
        return {
            "sp": 0.9+0.3*self.rng.random(),
            "se": 0.9+0.3*self.rng.random(),
            "h": self.rng.random()
        }

    def _regen_rules(self):
        n = self.prm.types
        F = np.zeros((n,n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                base = (self.rng.random()*2-1)*self.prm.frange + self.prm.bias
                sym = F[j,i] if j<i else base
                F[i,j] = max(-2.5, min(2.5, 0.6*base + 0.4*sym))
        return F

    def regen_rules(self):
        self.F = self._regen_rules()

    def reset_agents(self):
        self.A.clear()
        for i in range(self.prm.n):
            t = self.rng.randrange(self.prm.types)
            g = self._genome()
            self.A.append(Agent(
                x=self.rng.random()*self.width, y=self.rng.random()*self.height,
                vx=(self.rng.random()*4-2), vy=(self.rng.random()*4-2),
                t=t, age=0, e=40+self.rng.random()*60,
                hue=g["h"], sp=g["sp"], se=g["se"]
            ))

    def set_counts(self, n: int, types: int):
        self.prm.n = n
        if types != self.prm.types:
            self.prm.types = types
            self.R = np.array([self.prm.radius*(0.8+0.5*self.np_rng.rand()) for _ in range(types)], dtype=np.float32)
            for a in self.A:
                a.t = a.t % types
        # adjust population
        if len(self.A) < n:
            for _ in range(n-len(self.A)):
                t = self.rng.randrange(self.prm.types)
                g = self._genome()
                self.A.append(Agent(x=self.rng.random()*self.width, y=self.rng.random()*self.height,
                    vx=(self.rng.random()*4-2), vy=(self.rng.random()*4-2),
                    t=t, age=0, e=40+self.rng.random()*60,
                    hue=g["h"], sp=g["sp"], se=g["se"]))
        elif len(self.A) > n:
            del self.A[n:]

    def step(self):
        prm = self.prm
        self.fields.step(prm.reg)
        # spatial index
        self._clear_spatial()
        for i,a in enumerate(self.A):
            self._insert_spatial(i, a.x, a.y)

        tmp = []
        i = 0
        while i < len(self.A):
            a = self.A[i]
            rad = float(self.R[a.t])
            r2 = rad*rad
            fx = fy = 0.0

            ns = self._neighbors(a.x, a.y, rad, tmp)
            for j in ns:
                if j==i: continue
                b = self.A[j]
                dx = b.x - a.x; dy = b.y - a.y
                d2 = dx*dx + dy*dy
                if d2<=0.0 or d2>=r2: continue
                d = math.sqrt(d2)
                invd = 1.0/d
                q = d/rad
                s = float(self.F[a.t, b.t])*prm.fmul
                f = (-1.0 if q<0.5 else 1.0) * s * (1.0 - q*q)
                fx += f*dx*invd; fy += f*dy*invd

            # scent gradient
            gx = self.fields.sample(self.fields.scent_food, a.x+CELL, a.y) - self.fields.sample(self.fields.scent_food, a.x-CELL, a.y)                  - 0.8*(self.fields.sample(self.fields.scent_danger, a.x+CELL, a.y) - self.fields.sample(self.fields.scent_danger, a.x-CELL, a.y))
            gy = self.fields.sample(self.fields.scent_food, a.x, a.y+CELL) - self.fields.sample(self.fields.scent_food, a.x, a.y-CELL)                  - 0.8*(self.fields.sample(self.fields.scent_danger, a.x, a.y+CELL) - self.fields.sample(self.fields.scent_danger, a.x, a.y-CELL))
            fx += 0.07*a.se*gx; fy += 0.07*a.se*gy

            slow = 1.0 - 0.4*self.fields.sample(self.fields.terrain, a.x, a.y)

            a.vx = (a.vx + fx*prm.dt) * (prm.fric*slow)
            a.vy = (a.vy + fy*prm.dt) * (prm.fric*slow)
            vmax = 2.4*a.sp
            sp2 = a.vx*a.vx + a.vy*a.vy
            if sp2 > vmax*vmax:
                s = vmax / math.sqrt(sp2)
                a.vx *= s; a.vy *= s

            a.x += a.vx; a.y += a.vy
            if a.x<0: a.x+=self.width
            elif a.x>=self.width: a.x-=self.width
            if a.y<0: a.y+=self.height
            elif a.y>=self.height: a.y-=self.height

            # eat
            fi = self.fields.sample(self.fields.food, a.x, a.y)
            if fi > 0.02:
                bite = min(0.5*prm.dt, fi)
                self.fields.write(self.fields.food, a.x, a.y, fi-bite)
                a.e += 20.0*bite
                self.fields.add(self.fields.scent_food, a.x, a.y, -0.02)

            a.e -= 0.3 + 0.01*math.sqrt(sp2) + 0.015*rad/(prm.radius or 1.0)
            a.age += 1

            self.fields.add(self.fields.scent_food, a.x, a.y, 0.004)
            if a.e<20 or a.age>int(0.8*prm.maxage):
                self.fields.add(self.fields.scent_danger, a.x, a.y, 0.005)

            # reproduce
            if a.e>prm.birth and a.age>200:
                a.e *= 0.55
                child_t = a.t if self.rng.random()<0.94 else (a.t + (1 if self.rng.random()<0.5 else -1)) % prm.types
                hb = (a.hue + (self.rng.random()*0.16-0.08)) % 1.0
                self.A.append(Agent(
                    x=a.x + (self.rng.random()*6-3), y=a.y + (self.rng.random()*6-3),
                    vx=a.vx*0.15 + (self.rng.random()*2-1), vy=a.vy*0.15 + (self.rng.random()*2-1),
                    t=child_t, age=0, e=a.e*0.6, hue=hb,
                    sp=max(0.7, min(1.4, a.sp + (self.rng.random()*0.16-0.08))),
                    se=max(0.7, min(1.6, a.se + (self.rng.random()*0.16-0.08)))
                ))

            # die
            if a.e<=0 or a.age>prm.maxage:
                self.fields.add(self.fields.food, a.x, a.y, 0.2)
                self.fields.add(self.fields.scent_danger, a.x, a.y, 0.5)
                self.A[i] = self.A[-1]
                self.A.pop()
                self.nexts.pop()
                continue
            i += 1
