import logging


# class InMemoryLogHandler(logging.Handler):
#     def __init__(self):
#         super().__init__()
#         self.log_records = []
#
#     def emit(self, record):
#         log_entry = self.format(record)
#         self.log_records.append(log_entry)


# class StreamlitLogHandler(logging.Handler):
#     def __init__(self, widget_update_func):
#         super().__init__()
#         self.widget_update_func = widget_update_func
#
#     def emit(self, record):
#         msg = self.format(record)
#         self.widget_update_func(msg)

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
        self.log_area.markdown(f"```"
                               f"{msg_accumulated_in_buffer}"
                               f"```")

    def clear_logs(self):
        self.log_area.empty()  # Clear previous logs
