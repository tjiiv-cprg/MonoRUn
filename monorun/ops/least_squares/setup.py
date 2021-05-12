import os
from cffi import FFI


if __name__ == "__main__":
    os.system(
        'gcc -shared src/pnp_uncert_cpu.cpp -c '
        '-o src/pnp_uncert_cpu.cpp.o '
        '-fopenmp -fPIC -O2 -std=c++11 '
        '-I $CERES_INCLUDE_DIRS -I $Ceres_DIR/config -I /usr/include/eigen3')

    ffibuilder = FFI()
    with open(os.path.join(os.path.dirname(__file__), "src/ext.h")) as f:
        ffibuilder.cdef(f.read())
    ffibuilder.set_source(
        "_ext",
        """
        #include "src/ext.h"
        """,
        extra_objects=['src/pnp_uncert_cpu.cpp.o',
                       os.getenv('Ceres_DIR') + '/lib/libceres.so'],
        libraries=['stdc++', 'glog']
    )
    ffibuilder.compile(verbose=True)

    os.system("rm src/*.o")
    os.system("rm *.o")
