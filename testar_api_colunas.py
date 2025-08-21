import requests
import pandas as pd

API_KEY = "07062d8b-cbd2-4b81-addf-7bcce2bf1359"

# Endpoint com os parâmetros
URL = "https://sesgo.api.neovero.com/api/queries/execute/consulta_os_sesgo?data_abertura_inicio=2025-01-01T00:00&situacao_int=1,2,3"

# Cabeçalhos com autenticação correta
HEADERS = {
    "X-API-KEY": API_KEY
}

# Envia a requisição
response = requests.get(URL, headers=HEADERS)
response.raise_for_status()

# Converte para DataFrame e exibe somente o cabeçalho
df = pd.DataFrame(response.json())
print("✅ Colunas disponíveis:")
for col in df.columns:
    print(f"- {col}")
