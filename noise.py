import math


#  Utility functions
def hash32(x: int) -> int:
    """Simple 32-bit integer hash (deterministic pseudo-random)."""
    x &= 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7feb352d) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846ca68b) & 0xFFFFFFFF
    x ^= x >> 16
    return x & 0xFFFFFFFF


def hash2d(x: int, y: int, seed: int) -> int:
    """Hash integer coordinates (x, y) with a seed into a 32-bit integer."""
    h = (x + seed * 0x9e3779b1) & 0xFFFFFFFF
    k = (y ^ (seed * 0x85ebca6b)) & 0xFFFFFFFF
    h = hash32(h)
    k = hash32(k)
    return hash32(h ^ k)


def to_unit_double(v: int) -> float:
    """Map uint32 [0, 2^32-1] -> float [0,1]."""
    return v / 0xFFFFFFFF


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b."""
    return a + t * (b - a)


def fade(t: float) -> float:
    """Smooth fade function used in Perlin/value noise."""
    # 6t^5 - 15t^4 + 10t^3
    return t * t * t * (t * (t * 6 - 15) + 10)


#  Value noise
class ValueNoise2D:
    """Value noise: random value per grid point, interpolated."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def _lattice_value(self, ix: int, iy: int) -> float:
        h = hash2d(ix, iy, self.seed)
        return to_unit_double(h)  # [0,1]

    def sample(self, x: float, y: float) -> float:
        x0 = math.floor(x)
        y0 = math.floor(y)
        x1 = x0 + 1
        y1 = y0 + 1

        fx = x - x0
        fy = y - y0

        v00 = self._lattice_value(x0, y0)
        v10 = self._lattice_value(x1, y0)
        v01 = self._lattice_value(x0, y1)
        v11 = self._lattice_value(x1, y1)

        u = fade(fx)
        v = fade(fy)

        vx0 = lerp(v00, v10, u)
        vx1 = lerp(v01, v11, u)
        value = lerp(vx0, vx1, v)

        return value  # [0,1]

#  fBm Value Noise wrapper
class FBMValueNoise2D:
    """
    fBm (fractal Brownian motion) using ValueNoise2D as the base noise.
    """

    def __init__(
        self,
        seed: int = 0,
        octaves: int = 5,
        lacunarity: float = 2.0,
        gain: float = 0.5,
    ) -> None:
        self.base = ValueNoise2D(seed=seed)
        self.octaves = octaves
        self.lacunarity = lacunarity
        self.gain = gain

    def sample(self, x: float, y: float) -> float:
        amp = 1.0
        freq = 1.0
        total = 0.0
        amp_sum = 0.0

        for _ in range(self.octaves):
            total += self.base.sample(x * freq, y * freq) * amp
            amp_sum += amp

            freq *= self.lacunarity
            amp *= self.gain

        # normalize to [0,1]
        return total / amp_sum

#  Perlin noise (gradient)
class PerlinNoise2D:
    """Perlin-like gradient noise, normalized to [0,1]."""

    _gradients = [
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (0.70710678, 0.70710678),
        (-0.70710678, 0.70710678),
        (0.70710678, -0.70710678),
        (-0.70710678, -0.70710678),
    ]

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def _gradient(self, ix: int, iy: int):
        h = hash2d(ix, iy, self.seed)
        idx = h & 7  # 0..7
        return self._gradients[idx]

    def _dot_grid_gradient(self, ix: int, iy: int, x: float, y: float) -> float:
        gx, gy = self._gradient(ix, iy)
        dx = x - ix
        dy = y - iy
        return gx * dx + gy * dy

    def sample(self, x: float, y: float) -> float:
        x0 = math.floor(x)
        y0 = math.floor(y)
        x1 = x0 + 1
        y1 = y0 + 1

        sx = fade(x - x0)
        sy = fade(y - y0)

        n00 = self._dot_grid_gradient(x0, y0, x, y)
        n10 = self._dot_grid_gradient(x1, y0, x, y)
        n01 = self._dot_grid_gradient(x0, y1, x, y)
        n11 = self._dot_grid_gradient(x1, y1, x, y)

        ix0 = lerp(n00, n10, sx)
        ix1 = lerp(n01, n11, sx)
        value = lerp(ix0, ix1, sy)

        # Map roughly from [-1,1] to [0,1]
        v = value * 0.5 + 0.5
        # clamp just in case
        return max(0.0, min(1.0, v))


#  Worley / cellular noise
class WorleyNoise2D:
    """Worley / cellular noise: distance to nearest feature point."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def _feature_point(self, ix: int, iy: int):
        h = hash2d(ix, iy, self.seed)
        hx = hash32(h)
        hy = hash32(h ^ 0x68BC21EB)

        fx = to_unit_double(hx)
        fy = to_unit_double(hy)
        return ix + fx, iy + fy

    def sample(self, x: float, y: float) -> float:
        x_int = math.floor(x)
        y_int = math.floor(y)

        min_dist2 = 1e9

        for j in range(-1, 2):
            for i in range(-1, 2):
                cx = x_int + i
                cy = y_int + j
                px, py = self._feature_point(cx, cy)

                dx = px - x
                dy = py - y
                d2 = dx * dx + dy * dy
                if d2 < min_dist2:
                    min_dist2 = d2

        d = math.sqrt(min_dist2)
        # Normalize to ~[0,1]
        v = d / math.sqrt(3.0)
        v = max(0.0, min(1.0, v))
        # optionally invert:
        # v = 1.0 - v
        return v


#  Image export
def save_noise_image(noise, width: int, height: int, scale: float, filename: str) -> None:
    """
    Generate grayscale PGM image for given noise model.
    noise must have method sample(x, y) -> [0,1].
    """
    with open(filename, "wb") as f:
        header = f"P5\n{width} {height}\n255\n"
        f.write(header.encode("ascii"))

        for y in range(height):
            for x in range(width):
                nx = x * scale
                ny = y * scale
                v = noise.sample(nx, ny)
                v = max(0.0, min(1.0, v))
                f.write(bytes([int(v * 255)]))


def main():
    WIDTH = 512
    HEIGHT = 512
    SCALE = 0.02
    SEED = 1234

    value_noise = ValueNoise2D(seed=SEED)
    perlin_noise = PerlinNoise2D(seed=SEED)
    worley_noise = WorleyNoise2D(seed=SEED)

    # NEW: fBm value noise
    fbm_value_noise = FBMValueNoise2D(seed=SEED, octaves=6, lacunarity=2.0, gain=0.5)

    save_noise_image(value_noise, WIDTH, HEIGHT, SCALE, "output/value_noise.pgm")
    save_noise_image(fbm_value_noise, WIDTH, HEIGHT, SCALE, "output/fbm_value_noise.pgm")
    save_noise_image(perlin_noise, WIDTH, HEIGHT, SCALE, "output/perlin_noise.pgm")
    save_noise_image(worley_noise, WIDTH, HEIGHT, SCALE, "output/worley_noise.pgm")

    print("Images generated:\n"
          " - value_noise.pgm\n"
          " - fbm_value_noise.pgm\n"
          " - perlin_noise.pgm\n"
          " - worley_noise.pgm")
if __name__ == "__main__":
    main()
