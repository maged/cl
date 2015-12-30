package cl

// #cgo CFLAGS: -I/home/pblaberge/altera/14.0/hld/host/include
// #cgo LDFLAGS: -L/home/pblaberge/Downloads/arrow_c5sockit_bsp/arm32/lib -L/home/pblaberge/altera/14.0/hld/host/arm32/lib -L/home/pblaberge/altera/14.0/hld/host/arm32/lib -lalteracl -ldl -lacl_emulator_kernel_rt  -lalterahalmmd -lalterammdpcie -lelf -lrt -lstdc++
// #include "CL/opencl.h"
import "C"

import "errors"

var ErrUnsupported = errors.New("cl: unsupported")
