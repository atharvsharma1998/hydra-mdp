from enum import IntEnum


class classproperty(object):
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


class SceneFrameType(IntEnum):
    """Intenum for scene frame types."""

    ORIGINAL = 0
    SYNTHETIC = 1


class StateSE2Index(IntEnum):
    """Intenum for SE(2) arrays."""

    _X = 0
    _Y = 1
    _HEADING = 2

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_") and not attribute.startswith("__") and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classproperty
    def X(cls):
        return cls._X

    @classproperty
    def Y(cls):
        return cls._Y

    @classproperty
    def HEADING(cls):
        return cls._HEADING

    @classproperty
    def POINT(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classproperty
    def STATE_SE2(cls):
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)


class BoundingBoxIndex(IntEnum):
    """Intenum of bounding boxes in logs."""

    _X = 0
    _Y = 1
    _Z = 2
    _LENGTH = 3
    _WIDTH = 4
    _HEIGHT = 5
    _HEADING = 6

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_") and not attribute.startswith("__") and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classproperty
    def X(cls):
        return cls._X

    @classproperty
    def Y(cls):
        return cls._Y

    @classproperty
    def Z(cls):
        return cls._Z

    @classproperty
    def LENGTH(cls):
        return cls._LENGTH

    @classproperty
    def WIDTH(cls):
        return cls._WIDTH

    @classproperty
    def HEIGHT(cls):
        return cls._HEIGHT

    @classproperty
    def HEADING(cls):
        return cls._HEADING

    @classproperty
    def POINT2D(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classproperty
    def POSITION(cls):
        # assumes X, Y, Z have subsequent indices
        return slice(cls._X, cls._Z + 1)

    @classproperty
    def DIMENSION(cls):
        # assumes LENGTH, WIDTH, HEIGHT have subsequent indices
        return slice(cls._LENGTH, cls._HEIGHT + 1)


class LidarIndex(IntEnum):
    """Intenum for lidar point cloud arrays."""

    _X = 0
    _Y = 1
    _Z = 2
    _INTENSITY = 3
    _RING = 4
    _ID = 5

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_") and not attribute.startswith("__") and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classproperty
    def X(cls):
        return cls._X

    @classproperty
    def Y(cls):
        return cls._Y

    @classproperty
    def Z(cls):
        return cls._Z

    @classproperty
    def INTENSITY(cls):
        return cls._INTENSITY

    @classproperty
    def RING(cls):
        return cls._RING

    @classproperty
    def ID(cls):
        return cls._ID

    @classproperty
    def POINT2D(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classproperty
    def POSITION(cls):
        # assumes X, Y, Z have subsequent indices
        return slice(cls._X, cls._Z + 1)
