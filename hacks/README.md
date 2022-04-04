
fmha_to_global_memory.patch - patch for apex HEAD 23cfb57638497fdd1f2d8c09728b439b0e83efde,
replace shmem access to global memory access

trying to port fmha cheaply for deeper seqlen

results:
forward is slightly faster than py-version
backward is by 2x(!!) than py-version
