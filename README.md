# Pos_Artigo

## Treino de Rede Neural Convolucional 
Rede com objetivo de avaliar imagens de raio-x de torax, para classificar possivel incidencia de pneumunia. 

### Vesão do python usado no treino
- Python 3.13.3

### Arquivo com dependências da aplicação 
- requirements.txt 

## Dataset encontrado em:
https://data.mendeley.com/datasets/rscbjbr9sj/2?__hstc=25856994.84ad97dd589403f9105572a3d79fe02c.1747151584931.1747151584931.1747151584931.1&__hssc=25856994.2.1747151584932&__hsfp=1556678618

### OBS:
#### Treino executado com cuda, caso execute para atualização há verificação se pode ser executado utilizando GPU, caso não seja possivel executará com CPU. Dependendo de rodar em CPU ou GPU avaliar as variaveis:
- num_workers
- batch_size
