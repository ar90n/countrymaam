package common

import "runtime"

func GetProcNum(maxGoRoutines uint) uint {
	if maxGoRoutines == 0 {
		return uint(runtime.NumCPU())
	}

	return maxGoRoutines
}
