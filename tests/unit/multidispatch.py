from numbers import Number

from plum import dispatch


@dispatch
def f(x: str):
    return "This is a string!"


@dispatch
def f(x: int):
    return "This is an integer!"
#
#
@dispatch
def f(x: Number):
    return "This is a general number, but I don't know which type."
#

print(f("1"))

print(f(1))

print(f(1.0))
