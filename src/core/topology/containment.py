import numpy as np

class S3Containment:
    def __init__(self, matrix_27):
        self.matrix = matrix_27 # Матрица 27x27
        self.radius = self.calculate_radius()

    def calculate_radius(self):
        # Математика: R = sqrt(Tr(M* M))
        return np.sqrt(np.trace(self.matrix.T @ self.matrix))
