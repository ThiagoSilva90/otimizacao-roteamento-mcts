# Otimização de Roteamento de Veículos e Alocação de Cargas usando MCTS

Sistema computacional desenvolvido para a otimização de operações de distribuição em ambiente simulado. 

## Sequência de Execução
1. **Otimização de Rota:** O algoritmo Monte Carlo Tree Search (MCTS) determina a melhor sequência de entregas.
2. **Alocação de Veículo:** O sistema analisa a rota e a carga consolidada para selecionar o veículo mais adequado da frota.

## Estrutura do Projeto
- `src/models/`: Representação de Pedidos, Veículos e Pontos de Entrega.
- `src/algorithms/`: Implementação do algoritmo MCTS.
- `src/simulation/`: Ambiente simulado para avaliação de tempo e custo.