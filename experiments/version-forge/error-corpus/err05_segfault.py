# SIGSEGV: wild store through a null pointer (no traceback, signal death)
import ctypes
ctypes.c_int32.from_address(0).value = 1
