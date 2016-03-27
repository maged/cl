package cl

import "C"

type NaCL struct {
	Context *Context
	Queue   *CommandQueue
}

func NewNaCL() *NaCL {
	platforms, err := GetPlatforms()
	if err != nil {
		panic(err)
	}

	devices, err := GetDevices(platforms[0], DeviceTypeAll)
	if err != nil {
		panic(err)
	}

	context, err := CreateContext(devices[:1])
	if err != nil {
		panic(err)
	}
	commandQueue, err := context.CreateCommandQueue(devices[0], CommandQueueProfilingEnable)
	if err != nil {
		panic(err)
	}

	return &NaCL{Context: context, Queue: commandQueue}

}
