_name_to_channel_config = {}

def register_channel_config(cls):
    _name_to_channel_config[cls.__name__.lower()] = cls
    return cls

def name_to_channel_config(name):
    return _name_to_channel_config[name.lower()]

def is_channel_config(name):
    return name.lower() in _name_to_channel_config.keys()

def list_channel_config():
    return list(_name_to_channel_config.keys())