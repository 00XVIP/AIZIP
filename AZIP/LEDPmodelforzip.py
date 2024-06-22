import math
import numpy as np

class LEDPModel:
    def __init__(self):
        self.reset_training()

    def reset_training(self):
        self.m = 0
        self.m2 = 0
        self.m3 = 0
        self.m4 = 0
        self.m5 = 0
        self.m6 = 0
        self.b = 0
        self.m12 = 0
        self.m22 = 0
        self.m32 = 0
        self.m42 = 0
        self.m52 = 0
        self.m62 = 0
        self.best_params = None
        self.best_tolerance = float('inf')

    def build_model(self, DataX, Data_X2, DataY, Len_gen, True_zip, Epochs, LearningRate):
        self.Data_x = np.array(DataX)
        self.Data_x2 = np.array(Data_X2)
        self.Data_y = np.array(DataY)
        self.len_gen = Len_gen
        self.true_zip = True_zip
        self.epochs = Epochs
        self.learning_rate = LearningRate

        self.b_max = int(self.safe_compute(self.plus_test()))
        self.m_max = int(self.safe_compute(self.multiply_test()))
        self.m2_max = int(self.safe_compute(self.pow2_test()))
        self.m3_max = int(self.safe_compute(self.pow3_test()))
        self.m4_max = int(self.safe_compute(self.cos_test()))
        self.m5_max = int(self.safe_compute(self.sin_test()))
        self.m6_max = int(self.safe_compute(self.factoria_test()))
        self.m12_max = int(self.safe_compute(self.multiply_test2()))
        self.m22_max = int(self.safe_compute(self.pow2_test2()))
        self.m32_max = int(self.safe_compute(self.pow3_test2()))
        self.m42_max = int(self.safe_compute(self.cos_test2()))
        self.m52_max = int(self.safe_compute(self.sin_test2()))
        self.m62_max = int(self.safe_compute(self.factoria_test2()))

        self.b = -self.b_max
        self.m = -self.m_max
        self.m2 = -self.m2_max
        self.m3 = -self.m3_max
        self.m4 = -self.m4_max
        self.m5 = -self.m5_max
        self.m6 = -self.m6_max
        self.m12 = -self.m12_max
        self.m22 = -self.m22_max
        self.m32 = -self.m32_max
        self.m42 = -self.m42_max
        self.m52 = -self.m52_max
        self.m62 = -self.m62_max

        self.save()

    def safe_compute(self, value):
        return value if np.isfinite(value) else 0

    def save(self):
        self.best_params = {
            'b': self.b,
            'm': self.m,
            'm2': self.m2,
            'm3': self.m3,
            'm4': self.m4,
            'm5': self.m5,
            'm6': self.m6,
            'm12': self.m12,
            'm22': self.m22,
            'm32': self.m32,
            'm42': self.m42,
            'm52': self.m52,
            'm62': self.m62
        }
        self.best_tolerance = self.tolerance_save()

    def best_save(self):
        if self.tolerance_save() < self.best_tolerance:
            self.save()

    def tolerance_save(self):
        ra = 0
        for i in range(self.len_gen):
            try:
                y_pred = int(
                    (self.m * self.Data_x[i]) +
                    (self.m2 * self.Data_x[i] ** 2) +
                    (self.m3 * self.Data_x[i] ** 3) +
                    (self.m4 * np.cos(self.Data_x[i])) +
                    (self.m5 * np.sin(self.Data_x[i])) +
                    (self.m6 * math.factorial(int(self.Data_x[i]))) +
                    (self.m12 * self.Data_x2[i]) +
                    (self.m22 * self.Data_x2[i] ** 2) +
                    (self.m32 * self.Data_x2[i] ** 3) +
                    (self.m42 * np.cos(self.Data_x2[i])) +
                    (self.m52 * np.sin(self.Data_x2[i])) +
                    (self.m62 * math.factorial(int(self.Data_x2[i]))) +
                    self.b
                )
                if chr(y_pred % 256) != chr(self.Data_y[i]):
                    ra += 1
            except:
                ra += 1
        return ra

    def convergence_ver_setting(self):
        if self.m > self.m_max:
            self.b += self.learning_rate
            self.m = -self.m_max
        if self.b > self.b_max:
            self.m2 += 1
            self.b = -self.b_max
        if self.m2 > self.m2_max:
            self.m3 += self.learning_rate
            self.m2 = -self.m2_max
        if self.m3 > self.m3_max:
            self.m4 += self.learning_rate
            self.m3 = -self.m3_max
        if self.m4 > self.m4_max:
            self.m5 += self.learning_rate
            self.m4 = -self.m4_max
        if self.m5 > self.m5_max:
            self.m6 += self.learning_rate
            self.m5 = -self.m5_max
        if self.m6 > self.m6_max:
            self.m12 += self.learning_rate
            self.m6 = -self.m5_max
        if self.m12 > self.m12_max:
            self.m22 += self.learning_rate
            self.m12 = -self.m12_max
        if self.m22 > self.m22_max:
            self.m32 += self.learning_rate
            self.m22 = -self.m22_max
        if self.m32 > self.m32_max:
            self.m42 += self.learning_rate
            self.m32 = -self.m32_max
        if self.m42 > self.m42_max:
            self.m52 += self.learning_rate
            self.m42 = -self.m42_max
        if self.m52 > self.m52_max:
            self.m62 += self.learning_rate
            self.m52 = -self.m52_max
        if self.m62 > self.m62_max:
            self.learning_rate += self.learning_rate
            self.m62 = -self.m62_max

    def plus_test(self):
        return np.mean(self.Data_y - self.Data_x)

    def multiply_test(self):
        return np.mean(self.Data_y / self.Data_x)

    def pow2_test(self):
        return np.mean(self.Data_y / self.Data_x ** 2)

    def pow3_test(self):
        return np.mean(self.Data_y / self.Data_x ** 3)

    def cos_test(self):
        return np.mean(self.Data_y / np.cos(self.Data_x))

    def sin_test(self):
        return np.mean(self.Data_y / np.sin(self.Data_x))

    def factoria_test(self):
        return np.mean(self.Data_y / np.array([math.factorial(int(x)) for x in self.Data_x]))

    def multiply_test2(self):
        return np.mean(self.Data_y / self.Data_x2)

    def pow2_test2(self):
        return np.mean(self.Data_y / self.Data_x2 ** 2)

    def pow3_test2(self):
        return np.mean(self.Data_y / self.Data_x2 ** 3)

    def cos_test2(self):
        return np.mean(self.Data_y / np.cos(self.Data_x2))

    def sin_test2(self):
        return np.mean(self.Data_y / np.sin(self.Data_x2))

    def factoria_test2(self):
        return np.mean(self.Data_y / np.array([math.factorial(int(x)) for x in self.Data_x2]))

    def training_model(self):
        for _ in range(self.epochs):
            self.best_save()
            self.convergence_ver_setting()

    def return_data(self):
        return list(self.best_params.values())
