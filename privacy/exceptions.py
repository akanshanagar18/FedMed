class PrivacyError(Exception):
    pass

class InvalidUpdateError(PrivacyError):
    pass

class AggregationError(PrivacyError):
    pass

class ShapeMismatchError(PrivacyError):
    pass

class ContextError(PrivacyError):
    pass

class EncryptionError(PrivacyError):
    pass

class DecryptionError(PrivacyError):
    pass