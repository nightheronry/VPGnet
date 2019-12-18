# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.8
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info
if version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_IPM', [dirname(__file__)])
        except ImportError:
            import _IPM
            return _IPM
        if fp is not None:
            try:
                _mod = imp.load_module('_IPM', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _IPM = swig_import_helper()
    del swig_import_helper
else:
    import _IPM
del version_info
try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.


def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr_nondynamic(self, class_type, name, static=1):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    if (not static):
        return object.__getattr__(self, name)
    else:
        raise AttributeError(name)

def _swig_getattr(self, class_type, name):
    return _swig_getattr_nondynamic(self, class_type, name, 0)


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object:
        pass
    _newclass = 0


class scale_xy(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, scale_xy, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, scale_xy, name)
    __repr__ = _swig_repr
    __swig_setmethods__["step_x"] = _IPM.scale_xy_step_x_set
    __swig_getmethods__["step_x"] = _IPM.scale_xy_step_x_get
    if _newclass:
        step_x = _swig_property(_IPM.scale_xy_step_x_get, _IPM.scale_xy_step_x_set)
    __swig_setmethods__["step_y"] = _IPM.scale_xy_step_y_set
    __swig_getmethods__["step_y"] = _IPM.scale_xy_step_y_get
    if _newclass:
        step_y = _swig_property(_IPM.scale_xy_step_y_get, _IPM.scale_xy_step_y_set)
    __swig_setmethods__["xfMax"] = _IPM.scale_xy_xfMax_set
    __swig_getmethods__["xfMax"] = _IPM.scale_xy_xfMax_get
    if _newclass:
        xfMax = _swig_property(_IPM.scale_xy_xfMax_get, _IPM.scale_xy_xfMax_set)
    __swig_setmethods__["xfMin"] = _IPM.scale_xy_xfMin_set
    __swig_getmethods__["xfMin"] = _IPM.scale_xy_xfMin_get
    if _newclass:
        xfMin = _swig_property(_IPM.scale_xy_xfMin_get, _IPM.scale_xy_xfMin_set)
    __swig_setmethods__["yfMax"] = _IPM.scale_xy_yfMax_set
    __swig_getmethods__["yfMax"] = _IPM.scale_xy_yfMax_get
    if _newclass:
        yfMax = _swig_property(_IPM.scale_xy_yfMax_get, _IPM.scale_xy_yfMax_set)
    __swig_setmethods__["yfMin"] = _IPM.scale_xy_yfMin_set
    __swig_getmethods__["yfMin"] = _IPM.scale_xy_yfMin_get
    if _newclass:
        yfMin = _swig_property(_IPM.scale_xy_yfMin_get, _IPM.scale_xy_yfMin_set)
    __swig_setmethods__["ipmWidth"] = _IPM.scale_xy_ipmWidth_set
    __swig_getmethods__["ipmWidth"] = _IPM.scale_xy_ipmWidth_get
    if _newclass:
        ipmWidth = _swig_property(_IPM.scale_xy_ipmWidth_get, _IPM.scale_xy_ipmWidth_set)
    __swig_setmethods__["ipmHeight"] = _IPM.scale_xy_ipmHeight_set
    __swig_getmethods__["ipmHeight"] = _IPM.scale_xy_ipmHeight_get
    if _newclass:
        ipmHeight = _swig_property(_IPM.scale_xy_ipmHeight_get, _IPM.scale_xy_ipmHeight_set)

    def __init__(self):
        this = _IPM.new_scale_xy()
        try:
            self.this.append(this)
        except Exception:
            self.this = this
    __swig_destroy__ = _IPM.delete_scale_xy
    __del__ = lambda self: None
scale_xy_swigregister = _IPM.scale_xy_swigregister
scale_xy_swigregister(scale_xy)


def points_image2ground(n, m):
    return _IPM.points_image2ground(n, m)
points_image2ground = _IPM.points_image2ground

def points_ipm2image(n, m):
    return _IPM.points_ipm2image(n, m)
points_ipm2image = _IPM.points_ipm2image

def image_ipm(input, output):
    return _IPM.image_ipm(input, output)
image_ipm = _IPM.image_ipm
# This file is compatible with both classic and new-style classes.


