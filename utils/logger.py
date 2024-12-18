import logging
import os
from datetime import datetime

# Nome do diretório para armazenar logs
LOG_DIR = os.path.join(os.getcwd(), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)  # Cria apenas o diretório

# Nome do arquivo de log com timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Configuração do logger
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] %(lineno)d %(name)s %(levelname)s - %(message)s',
    level=logging.INFO,
)

# Mensagem para confirmar a inicialização do log
logging.info(f"Logger configurado. Logs serão salvos em: {LOG_FILE_PATH}")
