from __future__ import annotations
import numpy as np

CELL = 12

class Fields:
    def __init__(self, width: int, height: int, rng: np.random.RandomState, food_density: float):
        self.width = width
        self.height = height
        self.cols = width // CELL + 4
        self.rows = height // CELL + 4
        self.rng = rng

        self.terrain = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.food = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.scent_food = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.scent_danger = np.zeros((self.rows, self.cols), dtype=np.float32)

        self._seed_fields(food_density)

        # diffusion kernel (5-point laplacian)
        self._lap = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)

    def _noise(self, x, y):
        # Simple value noise using deterministic RNG grid
        return self.rng.rand(*x.shape).astype(np.float32) * 2 - 1

    def _seed_fields(self, food_density: float):
        # Terrain as blurred random noise for a soft 0..1 slope
        h, w = self.rows, self.cols
        base = self.rng.rand(h, w).astype(np.float32)
        for _ in range(3):
            # box blur
            base = (np.roll(base,1,0)+np.roll(base,-1,0)+np.roll(base,1,1)+np.roll(base,-1,1)+base)/5.0
        base = (base - base.min())/(base.ptp()+1e-6)
        self.terrain[:] = base

        self.food[:] = (self.rng.rand(h, w) < food_density).astype(np.float32) * (0.5 + 0.5*self.rng.rand(h,w))

    def step(self, regrow_rate: float):
        # diffuse scents and regrow food
        # Use convolution via neighbor rolls for speed without external libs
        for scent, rate in ((self.scent_food, 0.15), (self.scent_danger, 0.14)):
            lap = (np.roll(scent,1,0)+np.roll(scent,-1,0)+np.roll(scent,1,1)+np.roll(scent,-1,1)-4*scent)
            scent[:] = np.clip(scent + rate*lap - 0.004*scent, 0.0, 10.0)

        fert = 1.0 - 0.5*self.terrain
        f = self.food
        self.food[:] = np.clip(f + regrow_rate*fert*f*(1.0 - f), 0.0, 1.0)

    # grid sampling helpers
    def sample(self, arr: np.ndarray, px: float, py: float) -> float:
        gx = int(px//CELL); gy = int(py//CELL)
        if gx<0 or gy<0 or gx>=self.cols or gy>=self.rows: return 0.0
        return float(arr[gy, gx])

    def write(self, arr: np.ndarray, px: float, py: float, v: float):
        gx = int(px//CELL); gy = int(py//CELL)
        if gx<0 or gy<0 or gx>=self.cols or gy>=self.rows: return
        arr[gy, gx] = v

    def add(self, arr: np.ndarray, px: float, py: float, dv: float):
        gx = int(px//CELL); gy = int(py//CELL)
        if gx<0 or gy<0 or gx>=self.cols or gy>=self.rows: return
        arr[gy, gx] += dv
