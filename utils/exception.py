import sys
from logger import logging


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Gera uma mensagem detalhada para o erro capturado.
    
    Args:
        error (Exception): A exceção que ocorreu.
        error_detail (sys): Detalhes do sistema capturados via sys.exc_info.
    
    Returns:
        str: Mensagem detalhada do erro.
    """
    try:
        _, _, exc_tb = error_detail.exc_info()
        if exc_tb is not None:
            filename = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            return (f"Error occurred in script [{filename}] "
                    f"at line [{line_number}] "
                    f"with message: [{str(error)}]")
        else:
            return f"Error occurred: {str(error)}"
    except Exception as e:
        logging.error(f"Failed to generate detailed error message: {e}")
        return str(error)


class CustomException(Exception):
    """
    Classe personalizada para capturar exceções com mensagens detalhadas.
    """
    def __init__(self, error_message: Exception, error_detail: sys):
        """
        Inicializa a exceção personalizada.

        Args:
            error_message (Exception): A exceção que ocorreu.
            error_detail (sys): Detalhes do sistema capturados via sys.exc_info.
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
    
    def __str__(self) -> str:
        """
        Representação em string da exceção.

        Returns:
            str: Mensagem detalhada da exceção.
        """
        return self.error_message
