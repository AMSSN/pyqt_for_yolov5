# -*- coding: utf-8 -*-
_global_dict = {}

def set_value(key,value):
    """ 定义一个全局变量 """
    global _global_dict
    _global_dict[key] = value


def get_value(key, defValue=None):
    try:
        return _global_dict[key]
    except KeyError:
        return defValue


"""
set_value：添加变量
get_value：获取变量
"""