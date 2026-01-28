import numpy as np

class S3Containment:
    """Модель семантического удержания S3/T4."""
    def __init__(self, matrix_27):
        self.matrix = matrix_27
        self.R = self.calculate_radius()

    def calculate_radius(self):
        return np.sqrt(np.trace(self.matrix.T @ self.matrix))

    def generate_chronon(self, theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, c, -s], [0, 0, s, c]])

    def check_homeostasis(self, force_t4):
        p_res = 1 / (self.R ** 2)
        return p_res > force_t4
