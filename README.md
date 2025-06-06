# 🧠 Projeto de Neuroevolution – Computação Bioinspirada

Este projeto é uma simulação visual e interativa que demonstra como algoritmos bioinspirados podem ser usados para **treinar redes neurais artificiais sem backpropagation**, utilizando um processo evolutivo semelhante à **seleção natural**.

---

## 🎯 Objetivo

Evoluir redes neurais simples para controlar agentes (bolinhas verdes) que devem **alcançar um alvo vermelho** no centro da tela. Ao longo de várias gerações, as bolinhas aprendem a se mover de forma mais eficiente em direção ao alvo.

---

## 🧬 Conceitos envolvidos

Este projeto é uma aplicação de **neuroevolution**, uma técnica de **computação bioinspirada** que combina:

- 🧠 **Redes Neurais Artificiais (MLP)** – usadas como “cérebro” dos agentes.
- 🌿 **Algoritmos Evolutivos** – inspirados na evolução biológica:
  - **Seleção natural**: apenas os agentes mais eficazes se reproduzem.
  - **Mutação genética**: pequenas mudanças aleatórias nos "genes" (pesos da rede).
  - **Hereditariedade**: os melhores passam suas redes adiante.

---

## 📺 O que você verá na simulação

- Um **alvo vermelho fixo no centro** da tela.
- Várias **bolinhas verdes (agentes)** tentando alcançar o alvo.
- Nas primeiras gerações: movimentos aleatórios.
- Com o tempo: comportamento cada vez mais eficaz e direcionado.

---

## 📦 Requisitos

- Python 3.7+
- [pygame](https://www.pygame.org/)  
- numpy

### Instale as dependências com:

```bash
pip install pygame numpy
