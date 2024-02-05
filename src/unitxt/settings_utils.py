import os


class Settings:
    _instance = None
    _settings = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __setattr__(self, key, value):
        if key.endswith("_key") or key in {"_instance", "_settings"}:
            raise AttributeError(f"Modifying '{key}' is not allowed.")

        self._settings[key] = value

    def __getattr__(self, key):
        if key.endswith("_key"):
            actual_key = key[:-4]  # Remove the "_key" suffix
            return self.environment_variable_key_name(actual_key)

        env_value = os.getenv(self.environment_variable_key_name(key))

        if env_value is not None:
            return env_value

        if key in self._settings:
            return self._settings[key]

        raise AttributeError(f"'{key}' not found")

    def environment_variable_key_name(self, key):
        return "UNITXT_" + key.upper()

    def get_all_environment_variables(self):
        return [
            self.environment_variable_key_name(key) for key in self._settings.keys()
        ]


def get_settings():
    return Settings()
