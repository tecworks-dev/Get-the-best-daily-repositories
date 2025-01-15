import os
import random
import time


def construir_lista_estacoes_meteorologicas():
    """
    Obtém os nomes das estações meteorológicas a partir de um arquivo e remove duplicatas.
    """
    with open('./data/amostra_44k.csv', 'r', encoding="utf-8") as arquivo:
        return list(set(linha.split(';')[0] for linha in arquivo if "#" not in linha))


def converter_bytes(num):
    """
    Converte bytes para um formato legível (ex: KB, MB, GB).
    """
    for unidade in ['B', 'KB', 'MB', 'GB']:
        if num < 1024:
            return f"{num:.1f} {unidade}"
        num /= 1024


def formatar_tempo_decorrido(segundos):
    """
    Formata o tempo decorrido de forma simplificada.
    """
    minutos, segundos = divmod(segundos, 60)
    return f"{int(minutos)}m {int(segundos)}s" if minutos else f"{segundos:.2f}s"


def gerar_dados_teste(num_registros):
    """
    Gera e escreve um arquivo com medições sintéticas de temperatura.
    """
    inicio_tempo = time.time()
    nomes_estacoes = construir_lista_estacoes_meteorologicas()
    estacoes_10k_max = random.choices(nomes_estacoes, k=10_000)
    arquivo_saida = f"./data/medicoes_{num_registros}.txt"
    tamanho_lote = 10_000  # Processamento em lotes

    print(f"Criando {arquivo_saida}...")

    try:
        with open(arquivo_saida, 'w', encoding="utf-8") as arquivo:
            for _ in range(num_registros // tamanho_lote):
                lote = random.choices(estacoes_10k_max, k=tamanho_lote)
                linhas = '\n'.join([f"{estacao};{random.uniform(-99.9, 99.9):.1f}" for estacao in lote])
                arquivo.write(linhas + '\n')

        tamanho_arquivo = os.path.getsize(arquivo_saida)
        print(f"Arquivo gerado: {arquivo_saida}")
        print(f"Tamanho final: {converter_bytes(tamanho_arquivo)}")
        print(f"Tempo decorrido: {formatar_tempo_decorrido(time.time() - inicio_tempo)}")
    except Exception as e:
        print("Erro ao criar o arquivo:", e)


if __name__ == "__main__":
    num_registros = 1_000_000_000  # Número de registros parametrizado
    gerar_dados_teste(num_registros)
