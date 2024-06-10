import logging


class StreamlitLogHandler(logging.Handler):
    # Initializes a custom log handler with a Streamlit container for displaying logs
    def __init__(self, container):
        super().__init__()
        self.container = container
        self.log_area = self.container.empty()
        self.log_buffer = []

    def emit(self, record):
        msg = self.format(record)
        self.log_buffer.append(msg)
        msg_accumulated_in_buffer = ""
        for msg_item in self.log_buffer:
            msg_accumulated_in_buffer += f"{msg_item} \n"
        self.log_area.markdown(f"```" f"{msg_accumulated_in_buffer}" f"```")

    def clear_logs(self):
        self.log_area.empty()  # Clear previous logs
