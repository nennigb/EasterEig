# -*- coding: utf-8 -*-
# This file is part of eastereig, a library to locate exceptional points
# and to reconstruct eigenvalues loci.

# Eastereig is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Eastereig is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with Eastereig.  If not, see <https://www.gnu.org/licenses/>.
"""
##Define an adapter design patern using object composition.

It makes easier the implementation of linear algebra operation without concrete
reference to a specific library.

Reused from :
https://github.com/faif/python-patterns/blob/master/patterns/structural/adapter.py
"""


class Adapter(object):
    """Adapts an object by replacing methods.

    Usage:
    dog = Dog()
    dog = Adapter(dog, make_noise=dog.bark)

    Attributes
    ----------
    obj: initial object type
        contains a link to the original object

    #FIXME if the obj called a copy, get back an obj instead of an Adapter
    """

    def __init__(self, obj, **adapted_methods):
        """We set the adapted methods in the object's dict."""
        self.obj = obj
        self.__dict__.update(adapted_methods)

    def __getattr__(self, attr):
        """All non-adapted calls are passed to the object."""
        return getattr(self.obj, attr)

    def original_dict(self):
        """Print original object dict."""
        return self.obj.__dict__


def adaptVec(obj, lib):
    """Create an Vector-adapter for the different linear algera libraries.

    Parameters
    ----------
    obj : objet
        The objet to adapter
    lib : string
        String that describes the library: {'petsc', 'numpy', 'scipysp'}

    Returns
    -------
    adapted : adapter
        The adapted object

    Remarks
    -------
    copy, dot, duplicate return the native object type
    set return an adatper object
    """
#    def try_method(obj,method):
#        """ test if the method is present in the object
#        """
#        try:
#            return getattr(obj,method)
#        except:
#            return None

    if lib == 'petsc':
        ADAPTER = {'duplicate': obj.duplicate,  # duplicate copy patern but not the values, use before set!
                   'dot': obj.__mul__,
                   'copy': obj.copy,
                   'set': obj.set}  # 'set':try_method(obj,'set')}

    elif lib == 'numpy':
        ADAPTER = {'duplicate': obj.copy,
                   'dot': obj.dot,
                   'copy': obj.copy,
                   'set': obj.fill}

    # With scipysp the vector are full numpy vector, fill don't belong to scipy !
    elif lib == 'scipysp':
        ADAPTER = {'duplicate': obj.copy,
                   'dot': obj.__mul__,
                   'set': obj.fill}  # fill is not a method of scipy.sparse (try_method(obj,'fill'))
    else:
        raise NotImplementedError('The library {} is not yet implemented'.format(lib))

    # create the adapted objet
    return Adapter(obj, **ADAPTER)


def adaptMat(obj, lib):
    """Create an Matrix-adapter for the different linear algera libraries.

    Parameters
    ----------
    obj : objet
        The objet to adapter.
    lib : string
        String that describes the library {'petsc', 'numpy', 'scipysp'}.

    Returns
    -------
    adapted : adapter
        the adapted object

    Remarks
    -------
    copy, dot, duplicate return the native object type
    set return an adatper object
    """
    if lib == 'petsc':
        ADAPTER = {'duplicate': obj.duplicate,  # duplicate copy patern but not the values, use before set !
                   'dot': obj.__mul__,
                   'copy': obj.copy,
                   }

    elif lib == 'numpy':
        ADAPTER = {'duplicate': obj.copy,
                   'dot': obj.dot,
                   'copy': obj.copy
                   }

    elif lib == 'scipysp':
        ADAPTER = {'duplicate': obj.copy,
                   'dot': obj.__mul__,
                   'copy': obj.copy
                   }  # FIXME with scipy we use full numpy vector, fill don't belong to scipy !
    else:
        raise NotImplementedError('The library {} is not yet implemented'.format(lib))

    # create the adapted objet
    return Adapter(obj, **ADAPTER)
