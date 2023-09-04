import random
import numpy as np
# 定义问题的参数
POPULATION_SIZE = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
ELITE_RATE = 0.1
TARGET = 1

# 定义染色体类
class Chromosome:
    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2

    def fitness(self):
        return self.x2

# 定义遗传算法类
class GeneticAlgorithm:
    def __init__(self):
        self.population = []
        self.elite = []

    def initialize(self):
        # 初始化种群
        for i in range(POPULATION_SIZE):
            x1 = random.random()
            x2 = 1 - x1
            chromosome = Chromosome(x1, x2)
            self.population.append(chromosome)

    def evaluate(self):
        # 计算适应度
        for chromosome in self.population:
            chromosome.fitness()

    def select(self):
        # 选择优秀的染色体
        self.elite = sorted(self.population, key=lambda chromosome: chromosome.fitness())[:int(ELITE_RATE * POPULATION_SIZE)]

    def crossover(self):
        # 交叉染色体
        for i in range(POPULATION_SIZE - len(self.elite)):
            parent1 = random.choice(self.elite)
            parent2 = random.choice(self.elite)
            x1 = (parent1.x1 + parent2.x1) / 2
            x2 = 1 - x1
            child = Chromosome(x1, x2)
            self.population[i] = child

    def mutate(self):
        # 变异染色体
        for i in range(POPULATION_SIZE - len(self.elite)):
            chromosome = self.population[i]
            if random.random() < MUTATION_RATE:
                if random.random() < 0.5:
                    chromosome.x1 += random.uniform(-0.1, 0.1)
                    chromosome.x1 = max(0, min(1, chromosome.x1)) # 保证在0到1之间
                    chromosome.x2 = 1 - chromosome.x1
                else:
                    chromosome.x2 += random.uniform(-0.1, 0.1)
                    chromosome.x2 = max(0, min(1, chromosome.x2)) # 保证在0到1之间
                    chromosome.x1 = 1 - chromosome.x2

    def run(self):
        self.initialize()
        for i in range(100):
            self.evaluate()
            self.select()
            self.crossover()
            self.mutate()

        # 找到最优解
        best_chromosome = min(self.population, key=lambda chromosome: chromosome.fitness())
        return best_chromosome.x1, best_chromosome.x2

# 测试遗传算法
if __name__ == '__main__':
    ga = GeneticAlgorithm()
    x1, x2 = ga.run()
    print(f'The optimal solution is x1={x1}, x2={x2}, x1+x2={x1+x2}')