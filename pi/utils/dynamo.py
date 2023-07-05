import pyframe_eval


class dynamo:
    def __init__(self, callback):
        self.callback = callback

    def __enter__(self):
        pyframe_eval.set_eval_frame(None, self.callback)

    def __exit__(self, type, value, traceback):
        pyframe_eval.set_eval_frame(None, None)
