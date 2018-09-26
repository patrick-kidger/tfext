class TerminatedBatchData(RuntimeError):
    """Raised when an instance of BatchData has been
    terminated and is then called again.
    """
    def __init__(self, msg=None):
        if msg is None:
            msg = 'BatchData has already been terminated.'
        super(TerminatedBatchData, self).__init__(msg)
