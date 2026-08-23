import os
# load_dotenv lê o arquivo .env da raiz do projeto e coloca as variáveis
# dele (ex: ANTHROPIC_API_KEY) no ambiente do processo Python, como se
# você tivesse rodado "set ANTHROPIC_API_KEY=..." manualmente antes de
# executar o script.
from dotenv import load_dotenv
import anthropic

# Executa a leitura do .env assim que este arquivo é importado.
# Se o .env não existir, não dá erro — só não carrega nada (útil em
# ambientes onde a chave já vem de outra forma, ex: variável de ambiente
# do sistema operacional).
load_dotenv()

# Nome do modelo Claude usado em todo o módulo. Centralizado aqui pra
# trocar de modelo em um lugar só, se precisar no futuro.
# Claude Haiku 4.5: modelo mais barato e rápido da linha Claude —
# suficiente para gerar textos curtos de explicação/interpretação,
# sem precisar da capacidade (e do custo) de um modelo maior.
MODEL_NAME = "claude-haiku-4-5"


def get_client():
    """
    Cria e retorna um cliente autenticado da API da Anthropic.

    A chave de API é lida da variável de ambiente ANTHROPIC_API_KEY
    (carregada do .env pelo load_dotenv() acima). Nunca colocamos a
    chave direto no código — isso evitaria que ela vazasse se o
    repositório fosse compartilhado ou publicado.
    """
    # os.environ.get(...) busca a variável de ambiente; retorna None se
    # ela não existir, em vez de quebrar o programa na hora.
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    # Validação amigável: se a chave não foi configurada, avisamos
    # exatamente o que fazer, em vez de deixar o erro genérico da
    # biblioteca (menos claro) aparecer mais adiante.
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY não encontrada.\n"
            "Copie o arquivo .env.example para .env e cole sua chave de API "
            "(gerada em https://console.anthropic.com/settings/keys)."
        )

    # anthropic.Anthropic(...) é o cliente oficial da biblioteca "anthropic".
    # Ele guarda a chave e sabe como montar as requisições HTTP pra API.
    return anthropic.Anthropic(api_key=api_key)


def ask_claude(system_prompt, user_prompt, max_tokens=1024):
    """
    Envia UMA pergunta pro Claude e retorna a resposta em texto puro.

    Parâmetros:
      system_prompt: instruções de contexto/comportamento (ex: "você é um
                      assistente que ajuda médicos a interpretar exames").
                      Não é uma mensagem do usuário, é a "personalidade"
                      e as regras que o modelo deve seguir na conversa.
      user_prompt:    a pergunta/pedido específico dessa chamada
                      (ex: os dados de UM paciente + a predição do modelo).
      max_tokens:     limite de tamanho da resposta gerada, em tokens
                      (~4 caracteres cada). Evita respostas gigantes
                      desnecessárias e limita o custo da chamada.

    Retorna: uma string com o texto gerado pelo Claude.
    """
    client = get_client()

    try:
        # messages.create() é a chamada principal da API: envia o
        # system_prompt + a mensagem do usuário, e recebe a resposta.
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=max_tokens,
            system=system_prompt,
            # "messages" é uma lista porque a API suporta conversas com
            # várias trocas; aqui usamos só 1 mensagem (pergunta única,
            # sem histórico de conversa anterior).
            messages=[{"role": "user", "content": user_prompt}],
        )

    # Cada tipo de erro da API vira uma exceção própria na biblioteca —
    # tratamos separadamente pra dar uma mensagem clara sobre o que
    # aconteceu, em vez de um erro técnico genérico.
    except anthropic.AuthenticationError:
        raise ValueError(
            "Chave de API inválida. Confira o valor de ANTHROPIC_API_KEY no seu .env."
        )
    except anthropic.RateLimitError:
        raise RuntimeError(
            "Limite de requisições da API atingido. Aguarde um pouco e tente de novo."
        )
    except anthropic.APIStatusError as error:
        raise RuntimeError(f"Erro da API da Anthropic: {error.message}")

    # response.content é uma LISTA de blocos (a API pode devolver texto,
    # blocos de "pensamento", chamadas de ferramenta, etc). Como aqui só
    # pedimos uma resposta de texto simples, procuramos o primeiro bloco
    # do tipo "text" e pegamos o campo .text dele.
    text = next(
        (block.text for block in response.content if block.type == "text"),
        "",  # valor padrão caso, por algum motivo, não venha nenhum bloco de texto
    )

    return text
