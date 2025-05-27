import pygame
import random
import neat
import os
import math

# Configurações do jogo
largura_janela = 600
altura_janela = 600
tamanho_blocos = 20

# Inicialização do Pygame
pygame.init()
janela = pygame.display.set_mode((largura_janela, altura_janela))
pygame.display.set_caption("Snake AI - Neuroevolution")

# Cores
BRANCO = (255, 255, 255)
VERDE = (0, 255, 0)
VERMELHO = (255, 0, 0)
PRETO = (0, 0, 0)

# Classe da Cobra
class Cobra:
    def __init__(self):
        self.corpo = [[largura_janela // 2, altura_janela // 2]]
        self.direcao = [0, -tamanho_blocos]
        self.viva = True
        self.comida = self.gerar_comida()
        self.pontuacao = 0
        self.passos_sem_comer = 0

    def gerar_comida(self):
        while True:
            x = random.randrange(0, largura_janela, tamanho_blocos)
            y = random.randrange(0, altura_janela, tamanho_blocos)
            if [x, y] not in self.corpo:
                return [x, y]

    def mover(self):
        nova_cabeca = [self.corpo[0][0] + self.direcao[0], self.corpo[0][1] + self.direcao[1]]
        self.corpo.insert(0, nova_cabeca)
        if nova_cabeca == self.comida:
            self.pontuacao += 1
            self.comida = self.gerar_comida()
            self.passos_sem_comer = 0
        else:
            self.corpo.pop()
            self.passos_sem_comer += 1

        # Verifica colisões
        if (
            nova_cabeca in self.corpo[1:]
            or nova_cabeca[0] < 0
            or nova_cabeca[0] >= largura_janela
            or nova_cabeca[1] < 0
            or nova_cabeca[1] >= altura_janela
            or self.passos_sem_comer > 100
        ):
            self.viva = False

    def desenhar(self, janela):
        for segmento in self.corpo:
            pygame.draw.rect(janela, VERDE, (segmento[0], segmento[1], tamanho_blocos, tamanho_blocos))
        pygame.draw.rect(janela, VERMELHO, (self.comida[0], self.comida[1], tamanho_blocos, tamanho_blocos))

    def obter_entrada(self):
        cabeca = self.corpo[0]
        dx = self.comida[0] - cabeca[0]
        dy = self.comida[1] - cabeca[1]
        distancia_x = dx / largura_janela
        distancia_y = dy / altura_janela

        # Verifica obstáculos nas direções: frente, esquerda, direita
        esquerda = [self.direcao[1], -self.direcao[0]]
        direita = [-self.direcao[1], self.direcao[0]]

        def obstaculo(direcao):
            pos = [cabeca[0] + direcao[0], cabeca[1] + direcao[1]]
            if (
                pos in self.corpo
                or pos[0] < 0
                or pos[0] >= largura_janela
                or pos[1] < 0
                or pos[1] >= altura_janela
            ):
                return 1.0
            else:
                return 0.0

        entrada = [
            distancia_x,
            distancia_y,
            obstaculo(self.direcao),
            obstaculo(esquerda),
            obstaculo(direita),
            self.pontuacao / 10.0,
        ]
        return entrada

    def atualizar_direcao(self, acao):
        # Ação: 0 = frente, 1 = esquerda, 2 = direita
        esquerda = [self.direcao[1], -self.direcao[0]]
        direita = [-self.direcao[1], self.direcao[0]]
        if acao == 1:
            self.direcao = esquerda
        elif acao == 2:
            self.direcao = direita
        # Se acao == 0, mantém a direção atual

# Função principal para avaliação dos genomas
def avaliar_genomas(genomas, config):
    for genome_id, genome in genomas:
        rede = neat.nn.FeedForwardNetwork.create(genome, config)
        cobra = Cobra()
        relogio = pygame.time.Clock()
        tempo = 0

        while cobra.viva:
            pygame.event.pump()
            entrada = cobra.obter_entrada()
            saida = rede.activate(entrada)
            acao = saida.index(max(saida))
            cobra.atualizar_direcao(acao)
            cobra.mover()
            tempo += 1

            # Desenho do jogo
            janela.fill(PRETO)
            cobra.desenhar(janela)
            pygame.display.update()
            relogio.tick(15)

        # Avaliação da aptidão
        genome.fitness = cobra.pontuacao * 10 + tempo

# Função para executar o NEAT
def executar_neat(config_path):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    populacao = neat.Population(config)
    populacao.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    populacao.add_reporter(stats)

    populacao.run(avaliar_genomas, 50)

if __name__ == "__main__":
    caminho = os.path.dirname(__file__)
    caminho_config = os.path.join(caminho, "config-feedforward.txt")
    executar_neat(caminho_config)
