import random
import math
import numpy as np


def sphere_function(x):
    """Функція Сфери (Sphere function)"""
    return sum(xi ** 2 for xi in x)


def generate_random_point(bounds):
    """Генерує випадкову точку в заданих межах"""
    return [random.uniform(low, high) for low, high in bounds]


def get_neighbor(point, bounds, sigma=0.1):
    """Генерує сусідню точку з нормальним розподілом"""
    neighbor = []
    for i, (xi, (low, high)) in enumerate(zip(point, bounds)):
        # Використовуємо нормальний розподіл для генерації сусіда
        new_xi = xi + random.gauss(0, sigma)
        # Обмежуємо значення межами
        new_xi = max(low, min(high, new_xi))
        neighbor.append(new_xi)
    return neighbor


def hill_climbing(func, bounds, iterations=1000, epsilon=1e-6):
    """Алгоритм підйому на гору"""
    # Початкова випадкова точка
    current_point = generate_random_point(bounds)
    current_value = func(current_point)
    
    for _ in range(iterations):
        # Генеруємо сусідню точку
        neighbor_point = get_neighbor(current_point, bounds)
        neighbor_value = func(neighbor_point)
        
        # Якщо знайдено краще рішення - переміщуємося туди
        if neighbor_value < current_value:
            if abs(neighbor_value - current_value) < epsilon:
                break
            current_point = neighbor_point
            current_value = neighbor_value
            
    return current_point, current_value


def random_local_search(func, bounds, iterations=1000, epsilon=1e-6):
    """Випадковий локальний пошук"""
    best_point = generate_random_point(bounds)
    best_value = func(best_point)
    
    for _ in range(iterations):
        # Генеруємо нову випадкову точку
        new_point = generate_random_point(bounds)
        new_value = func(new_point)
        
        # Якщо знайдено краще рішення - зберігаємо його
        if new_value < best_value:
            if abs(new_value - best_value) < epsilon:
                break
            best_point = new_point
            best_value = new_value
            
    return best_point, best_value


def simulated_annealing(func, bounds, iterations=1000, temp=1000, cooling_rate=0.95, epsilon=1e-6):
    """Алгоритм імітації відпалу"""
    current_point = generate_random_point(bounds)
    current_value = func(current_point)
    best_point = current_point
    best_value = current_value
    temperature = temp
    
    for _ in range(iterations):
        if temperature < epsilon:
            break
            
        # Генеруємо сусідню точку
        neighbor_point = get_neighbor(current_point, bounds)
        neighbor_value = func(neighbor_point)
        
        # Обчислюємо різницю значень
        delta = neighbor_value - current_value
        
        # Приймаємо нове рішення відповідно до критерію Метрополіса
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_point = neighbor_point
            current_value = neighbor_value
            
            # Оновлюємо найкраще знайдене рішення
            if current_value < best_value:
                best_point = current_point
                best_value = current_value
        
        # Зменшуємо температуру
        temperature *= cooling_rate
        
    return best_point, best_value


if __name__ == "__main__":
    # Встановлюємо seed для відтворюваності результатів
    random.seed(42)
    np.random.seed(42)
    
    # Межі для функції (2-вимірний простір)
    bounds = [(-5, 5), (-5, 5)]
    
    # Параметри алгоритмів
    iterations = 1000
    epsilon = 1e-6
    
    # Виконання алгоритмів
    print("Hill Climbing:")
    hc_solution, hc_value = hill_climbing(sphere_function, bounds, iterations, epsilon)
    print(f"Розв'язок: {hc_solution}\nЗначення: {hc_value}")

    print("\nRandom Local Search:")
    rls_solution, rls_value = random_local_search(sphere_function, bounds, iterations, epsilon)
    print(f"Розв'язок: {rls_solution}\nЗначення: {rls_value}")

    print("\nSimulated Annealing:")
    sa_solution, sa_value = simulated_annealing(sphere_function, bounds, iterations, temp=1000, cooling_rate=0.95, epsilon=epsilon)
    print(f"Розв'язок: {sa_solution}\nЗначення: {sa_value}") 