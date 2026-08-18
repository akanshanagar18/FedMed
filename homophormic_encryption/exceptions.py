class PrivacyModuleError(Exception):
    pass

class ContextError(PrivacyModuleError):
    pass

class EncryptionError(PrivacyModuleError):
    pass

class DecryptionError(PrivacyModuleError):
    pass

class AggregationError(PrivacyModuleError):
    pass

class ShapeMismatchError(PrivacyModuleError):
    pass
