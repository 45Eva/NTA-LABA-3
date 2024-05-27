import math
from sympy import factorint, mod_inverse, Matrix
import numpy as np
import random
from itertools import combinations
from datetime import datetime
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed


def is_prime(num):
    """Перевірка на простоту числа."""
    if num <= 1:
        return False
    if num <= 3:
        return True
    if num % 2 == 0 or num % 3 == 0:
        return False
    i = 5
    while i * i <= num:
        if num % i == 0 or num % (i + 2) == 0:
            return False
        i += 6
    return True


def generate_prime_list(limit):
    """Генерація списку простих чисел до певного значення."""
    primes = []
    num = 2
    while num < limit:
        if is_prime(num):
            primes.append(num)
        num += 1
    return primes


def calculate_B(n, c=3.38):
    """Обчислення значення B."""
    log_n = math.log(n)
    log_log_n = math.log(log_n)
    B = c * math.exp(0.5 * (log_n * log_log_n) ** 0.5)
    return int(B)


def form_factor_base(n, c=3.38):
    """Формування факторної бази."""
    B = calculate_B(n, c)
    factor_base = generate_prime_list(B)
    return factor_base


def is_smooth(value, factor_base):
    """Перевірка на гладкість за факторною базою."""
    if value == 1:
        return False  # Число 1 не має простих дільників, тому воно не є гладким
    factors = factorint(value)
    return all(p in factor_base for p in factors)


def find_smooth_relation(alpha, n, factor_base):
    """Пошук одного співвідношення (рівняння)."""
    t = len(factor_base)
    while True:
        k = random.randint(0, n - 1)
        alpha_k = pow(alpha, k, n)

        if is_smooth(alpha_k, factor_base):
            factors = factorint(alpha_k)
            equation = [0] * (t + 1)
            equation[-1] = k  # Вільний член (правий бік рівняння)

            for i, p in enumerate(factor_base):
                if p in factors:
                    equation[i] = factors[p]

            return equation


# Налаштування логування
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def form_linear_equations(alpha, n, factor_base):
    """Формування лінійних рівнянь для дискретних логарифмів з розпаралелюванням."""
    t = len(factor_base)
    num_equations = t + 15
    equations = []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(find_smooth_relation, alpha, n, factor_base) for _ in range(num_equations)]

        for future in as_completed(futures):
            try:
                equation = future.result()
                equations.append(equation)
                # logging.info(f"Додано рівняння: {equation}")
            except Exception as e:
                logging.error(f"Помилка при виконанні завдання: {e}")

    return equations


def find_square_submatrix_modulo(A):
    """Знаходження квадратної підматриці, яка є невиродженою по модулю."""
    A = np.array(A)
    t = A.shape[1] - 1  # Розмір факторної бази
    for indices in combinations(range(A.shape[0]), t):
        submatrix = A[np.ix_(indices, range(t))]
        if np.linalg.matrix_rank(submatrix) == t:
            M = submatrix
            b = A[np.array(indices), -1]
            return M, b
    raise ValueError("Не вдалося знайти підхожу квадратну підматрицю з лінійно незалежними рядками.")


def modular_matrix_inverse(A, n):
    """Обчислення оберненої матриці в модульній арифметиці."""
    A = Matrix(A.tolist())
    try:
        A_inv = A.inv_mod(n - 1)
        return np.array(A_inv).astype(int)
    except:
        return None


def solve_linear_system(equations, n, alpha, factor_base):
    """Розв'язання системи лінійних рівнянь модульною арифметикою через обернену матрицю."""
    t = len(equations[0]) - 1
    while True:
        A = np.array([eq[:t] for eq in equations])
        b = np.array([eq[t] for eq in equations])
        M, b_vector = find_square_submatrix_modulo(np.column_stack((A, b)))
        M_inv = modular_matrix_inverse(M, n)
        if M_inv is not None:
            # print("Матриця M:")
            # print(M)
            # print("Матриця b:")
            # print(b_vector)
            # print("Обернена M:")
            # print(M_inv)
            x = np.dot(M_inv, b_vector) % (n - 1)
            x = [int(val) for val in x]
            break
        else:
            print("Матриця не має оберненої. Генеруємо нову матрицю.")
            equations = form_linear_equations(alpha, n, factor_base)
    return x


def compute_discrete_log(alpha, beta, factor_base, log_alpha_p, n):
    """Обчислення дискретного логарифму log_alpha(β)."""
    t = len(factor_base)
    l = 0

    while l < n:
        beta_alpha_l = (beta * pow(alpha, l, n)) % n

        if is_smooth(beta_alpha_l, factor_base):
            factors = factorint(beta_alpha_l)
            d = [0] * t
            for i, p in enumerate(factor_base):
                if p in factors:
                    d[i] = factors[p]

            log_beta = (sum(d[i] * log_alpha_p[i] for i in range(t)) - l) % (n - 1)
            return log_beta

        l += 1

    raise ValueError("Не вдалося знайти гладке число для обчислення дискретного логарифму.")


def main():
    # Введення параметрів користувачем
    alpha = int(input("Введіть генератор групи (α): "))
    beta = int(input("Введіть елемент групи (β): "))
    n = int(input("Введіть порядок групи (n): "))

    # Формування факторної бази
    factor_base = form_factor_base(n)
    # print("Факторна база:", factor_base)

    # Кількість необхідних рівнянь
    t = len(factor_base)

    # Формування лінійних рівнянь
    equations = form_linear_equations(alpha, n, factor_base)
    # print("Сформовані лінійні рівняння:")
    # for eq in equations:
    #    print(eq)

    # Розв'язання системи лінійних рівнянь
    log_alpha_p = solve_linear_system(equations, n, alpha, factor_base)
    print("Дискретні логарифми для факторної бази:")
    for p, log_p in zip(factor_base, log_alpha_p):
        print(f"log_{alpha}({p}) = {log_p}")

    start = datetime.now()
    # Обчислення дискретного логарифму log_alpha(β)
    log_beta = compute_discrete_log(alpha, beta, factor_base, log_alpha_p, n)
    stop = datetime.now()
    print(f"Час виконання алгоритму: {stop - start}")
    print(f"Дискретний логарифм log_{alpha}({beta}) = {log_beta}")


# Виконання головної функції
if __name__ == "__main__":
    main()


