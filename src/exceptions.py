class OpenQuantError(Exception):
    pass


class DataSourceError(OpenQuantError):
    pass


class ValidationError(OpenQuantError):
    pass


class ConfigError(OpenQuantError):
    pass
