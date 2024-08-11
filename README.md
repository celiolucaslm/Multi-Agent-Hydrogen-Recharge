 # Multi-Agent-Hydrogen-Recharge

 Nesse repositório foi implementado o ambiente de aprendizado multi-agente modelado para a recarga de hidrogênio através de estações móveis (veículos), descrito no [relatório de estágio]() (em francês).
 
 O diretório [env](https://github.com/celiolucaslm/Multi-Agent-Hydrogen-Recharge/tree/main/multi_hydrogen_recharge/env) contém a classe de implementação do ambiente MultiHydrogenRecharge. Enquanto no [multi_hydrogen_recharge](https://github.com/celiolucaslm/Multi-Agent-Hydrogen-Recharge/tree/main/multi_hydrogen_recharge) está contido as classes que são utilizadas pelo ambiente de aprendizado bem como o código para o treinamento do algoritmo [MADDPG](https://arxiv.org/pdf/1706.02275), além disso, os códigos de teste do algoritmo e para tomada de ações aleatórias utilizados para comparar os resultados também estão presentes. 

## Vídeo de exemplo do funcionamento do ambiente
 [Video](https://github.com/user-attachments/assets/e5bf792b-507e-4da9-a95d-fd3b35ca5d11)

 Nesse caso, os Veículos e Comandas estabelecem suas listas de preferência para que o algoritmo de matching realize a atribuição entre eles. O objetivo é fazer com que os Veículos aprendam a priorizar as Comandas que o desejam, a cada step dado o Véiculo muda para a posição da Comanda atendida e perde hidrogênio proporcionalmente ao tempo de serviço realizado.
