
#importar bibliotecas

import pygame #janela para o jogo
import numpy as np #importa biblioteca de numeros, vetores e matrizes
import random #random faz funcoes aleatorias, serve para escolher aleatoriamente pais, fazer mutacoes etc
import sys #fechar simulacao no x

# Configurações da janela, de simulacao do joguinho
WIDTH, HEIGHT = 600, 600
TARGET = np.array([WIDTH // 2, HEIGHT // 2])
AGENT_RADIUS = 5

## C

N_AGENTS = 50 #quantidade de agentes por geracao
N_GENERATIONS = 100 #quantidade de geracoes 
STEPS_PER_AGENT = 100 #cada bolinha pode se mover 100x
MUTATION_RATE = 0.1 #probabilidade de mudar aleatoriamente os genes (pesos da rede neural) ao gerar um novo agente. - diversidade

#inicializando a parte do joguinho
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Projeto Neuroevolution - computacao bioinspirada")

clock = pygame.time.Clock()

# funcao para medir distancia entre dois pontos, a e b posicao entre agente e o alvo.
#quanto mais proximo do alvo, melhor o agente
#retorna a distancia entre eles. Usamos isso na avaliacao da aptidão da bolinha
def distance(a, b):
    return np.linalg.norm(a - b)

# Classe da rede neural - codificacao genetica da solucao c/ 1 camada escondida

#aqui é criado o cerebro artificial de cada bolinha (Agente). Cria uma classe chamada neuralnetwork. É tipo
#um molde pra construir diversos cerebros artificiais
class NeuralNetwork:
    def __init__(self):
        self.w1 = np.random.randn(2, 5) #pesos da entrada para camada escondida
        self.b1 = np.random.randn(5) #bias da camada escondida
        self.w2 = np.random.randn(5, 2) #pesos da camada escondida para a saida
        self.b2 = np.random.randn(2) #bias da saida

        #esses sao os genes do agente. cada bolinha nasce com valores diferentes.
        #genes = determinam o comportamento e a performance. 

    def forward(self, x):
        h = np.tanh(np.dot(x, self.w1) + self.b1)
        out = np.tanh(np.dot(h, self.w2) + self.b2)
        return out

#clonagem - hereditariedade. Cria uma cópia da rede neural atual, para passar os genes adiante durante a reproducao
#equivale a filhos herdando o dna dos pais
    def clone(self):
        clone = NeuralNetwork()
        clone.w1 = self.w1.copy()
        clone.b1 = self.b1.copy()
        clone.w2 = self.w2.copy()
        clone.b2 = self.b2.copy()
        return clone

#mutacao genetica. Para manter diversidade na populacao, permitindo que agentes explorem novos comportamentos.
    def mutate(self):
        for param in [self.w1, self.b1, self.w2, self.b2]:
            mutation_mask = np.random.rand(*param.shape) < MUTATION_RATE
            param += mutation_mask * np.random.randn(*param.shape) #decide aleatoriamente quais genes serao alterados

# Agente controlado por rede neural
class Agent:
    def __init__(self):
        self.nn = NeuralNetwork()
        self.reset()

    def reset(self):
        self.pos = np.random.rand(2) * np.array([WIDTH, HEIGHT])
        self.traj = [self.pos.copy()]
        self.fitness = 0

    def update(self):
        direction = TARGET - self.pos
        direction /= np.linalg.norm(direction) + 1e-8
        move = self.nn.forward(direction)
        self.pos += move * 5
        self.pos = np.clip(self.pos, [0, 0], [WIDTH, HEIGHT])
        self.traj.append(self.pos.copy())

#funcao de avaliacao - fitness function 

    def evaluate(self):
        self.fitness = 1 / (distance(self.pos, TARGET) + 1) #mede o desempenho do agente. Quao eficaz ele fez a tarefa de chegar
        #no alvo = O agente que chegar mais perto do alvo tem maior fitness

#selecao natural + nova geracao

def next_generation(agents):
    agents.sort(key=lambda a: a.fitness, reverse=True) #organiza os agentes do melhor para o pior, de acordo com o fitness

    best = agents[:N_AGENTS//2] #selecao natural. os melhores vao reproduzi. seleciona a metade superior (melhores 50%)
    new_agents = []

    #reproducao - cruzamento genetico. Escolhe aleatoriamente um dos melhores agentes
    #cria um filho clonado, com o mesmo cerebro (rede neural)
    for _ in range(N_AGENTS):
        parent = random.choice(best)
        child = Agent()
        child.nn = parent.nn.clone()

        #mutacao genética - aplica pequenas mudancas aleatorias nos genes (pesos e bias da rede neural)

        child.nn.mutate()
        new_agents.append(child) #formacao da nova geracao

    return new_agents

# Criar uma população inicial:
agents = [Agent() for _ in range(N_AGENTS)]
generation = 0

while generation < N_GENERATIONS:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

#avaliacao e evolucao da geracao

    for agent in agents:
        agent.reset()

    for _ in range(STEPS_PER_AGENT):
        screen.fill((30, 30, 30))
        pygame.draw.circle(screen, (255, 0, 0), TARGET.astype(int), 8)

        for agent in agents:
            agent.update()
            pygame.draw.circle(screen, (0, 255, 0), agent.pos.astype(int), AGENT_RADIUS)

        pygame.display.flip()
        clock.tick(60)


    for agent in agents:
        agent.evaluate()


    best = max(agents, key=lambda a: a.fitness)
    print(f"Geração {generation} - Melhor distância: {1 / best.fitness:.2f}")

 
    agents = next_generation(agents)
    generation += 1
