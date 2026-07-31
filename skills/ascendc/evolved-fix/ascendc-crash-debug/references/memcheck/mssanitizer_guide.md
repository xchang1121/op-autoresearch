# MsSanitizer Memory Testing Tool Guide

> **Document version**: MindStudio 80. RC1
> **Applicable product**: Atlas series developers package/module group

## 1. Overview

Memory Check is an anomaly detection function for the user program runtime.**MsSanitizer**tool detects and reports unusual cross-border and unmatched memory access to external and internal storages in operator operations.

> **⚠ ️ Note:
> *   The msSanitizer tool**does not support**the memory testing of the operator warehouse in Ascend Transformer Boost.
> *   When users use framework, such as PyTorch, to access operator, framework's internal memory may be managed through a memory pool. The manual reporting interface (`SanitizerReportMalloc` / `SanitizerReportFree`) is used to ensure accuracy.

---

## 2. Supported memory anomaly type

Memory tests allow for the identification and reporting of the following six core types of anomalies:

| Abnormal Name | Description | Organisation | Supporting address space |
| :--- | :--- | :--- | :--- |
| **Illegally read and write**<br> (Illegal Read/Write) | Unallocated RAM areas were visited. | Kernel, Host | GM, UB, L0{A,B,C}, L1 |
| **Multi-nucleus**<br> (Multi-core Overwrite) | Several AI Cores have visited overlapping memory areas and at least one core has performed writing operations. | Kernel | GM |
| **Non-matched**<br> (Misaligned Access) | The address of the DMA moving data does not meet the minimum hardware access particle alignment requirement. | Kernel | GM, UB, L0{A,B,C}, L1 |
| **Unlawfully released**<br> (Illegal Free) | Attempt to release undistributed or released memory addresses. | Host | GM |
| **Memory leak**<br> (Memory Leak) | Not released after application of memory, resulting in continued increase in memory occupancy during operation. | Host | GM |
| **Distribution of unused memory**<br> (Unused Memoory) | The memory allocation was not accessed or used until the procedure was completed. | Kernel, Host | GM |

---

## 3. Enable memory detection

The base memory detection (memcheck) is enabled by default when running the `msSanitizer` tool.

### 3.1 Basic testing orders
The following are some of the most recent examples of illegal reading and writing, multi-nuclei, non-matching and illegal release tests:
```bash
mssanitizer --tool=memcheck <application>
```

### 3.2 Advanced testing options

*   **Commencing memory leak detection**:
    If memory leaks are to be detected, the `--leak-check=yes` parameter needs to be added visibly:
    ```bash
    mssanitizer --tool=memcheck --leak-check=yes <application>
    ```

*   **Turn on the distribution memory unused detection**:
    If you need to detect the assigned but not used memory, you need to add `--check-unused-memory=yes` parameters visibly:
    ```bash
    mssanitizer --tool=memcheck --check-unused-memory=yes <application>
    ```

> **💡 tip**:
> *   The anomaly report will be printed to the terminal after the user program has been run.
> *   The tool also supports the illegal reading and writing testing of HCCL communications interfaces (such as AllReduce, AllGather, etc.) and the functional integration class operator.

---

## 4. Memory anomaly resolution

The following is a typical format and interpretation of the various unusual reports.

### 4.1 Illegal reading and writing (Illegal Read/Write)
**Meaning**: operator visited undistributed GM or film memory (over and above hardware capacity).

**Example report**:
```text
====== ERROR : illegal read of size 224
====== at 0x12c0c0015000 on GM in add_custom_kernel
====== in block aiv(0) on device 0
====== code in pc current 0x77c (serialNo: 10)
====== #0  $ {ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl/dav_c220/kernel_operator_data_copy_impl.h:58:9
====== #1 ...
====== #3 illegal_read_and_write/add_custom.cpp:18:5
```


Interpret:

Error type: Illegally read 224 bytes.

Location: GM address 0x12c0c0015,000, in add_custom_kernel kernel.

Code location: Correlation source file add_custom.cpp Line 18.

Note: If the debugging option is not added, call stack information #0 to #3 may not be displayed.


### 4.2 Multi-nuclei

AI Core is the core of calculation in the Quest AI processor, where there are multiple AI Core, operator operations. These AI Cores move data from GM to or out of the calculation process. When there is no visible inter-nuclei synchronization, there is a problem of multi-touching if there is an overlap in the GM that is accessed between the cores and there is at least one cross-checking address to write. Here we make sure that there is no problem of stepping between the multiple cores through the owner 's concept, and when one memory is written into one, the memory is left with the other checking this memory.

**Example report:**

```text
====== WARNING : out of bounds of size 256
// The basic information of the anomaly, containing the number of bytes in which step was taken
====== at 0x12c0c00150fc on GM when writing data in add_custom_kernel
// An abnormal memory location information, including the kernel name, address space and memory address, where the memory address is the first address in a memory access
====== in block aiv(9) on device 0
// Anomalous code corresponds to the block index that vector has cored
====== code in pc current 0x7b8 (serialNo: 22)
// Current Abnormal Pc Pointer and Call api Behaviour Serial Number
====== #0  $ {ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl/dav_c220/kernel_operator_data_copy_impl.h:103:9
// call stack, with file names, rows and column numbers, with the following exception code:
====== #1  $ {ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/inner_interface/inner_kernel_operator_data_copy_intf.cppm:155:9
====== #2  $ {ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/inner_interface/inner_kernel_operator_data_copy_intf.cppm:461:5
====== #3 out_of_bound/add_custom.cpp:21:5
```
In the above example, 256 bytes of access took place, there was a multi-nuclear step during the visit to the "0x12c0c00150fc" address on GM, and the command leading to the anomaly corresponded to line 21 of operator to achieve the add_custom.cpp file.

### 4.3 Non-matched visits (Misaligned Access)

Meaning: Access to an address does not meet the minimum particle size requirements for DMA handling (e. g. 32 byte or 128 byte alignment), which may result in data errors or AI Core anomalies.

Example report:
```text
====== ERROR : misaligned access of size 13
// Basic information on anomalies, including bytes in which alignment abnormalities occur
====== at 0x6 on UB in add_custom_kernel
// Anecdotal memory location information, including kernel name, address space and memory address
====== in block aiv(0) on device 0
// Anomalous code corresponds to the block index that vector has cored
====== code in pc current 0x780 (serialNo: 33)
// Current Abnormal Pc Pointer and Call api Behaviour Serial Number
====== #0  $ {ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl/dav_c220/kernel_operator_data_copy_impl.h:103:9
// call stack, with file names, rows and column numbers, with the following exception code:
====== #1  $ {ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/inner_interface/inner_kernel_operator_data_copy_intf.cppm:155:9
====== #2  $ {ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/inner_interface/inner_kernel_operator_data_copy_intf.cppm:461:5
====== #3 illegal_align/add_custom.cpp:18:5
```
In the above example, there are 13 bytes of alignment abnormal visits, alignment problems at the "0x6" address on the UB, and the command leading to the anomaly corresponds to row 18 of operator to achieve the add_custom.cpp file.
Note: The exception report will not contain call stack information without adding the compiler option.

### 4.4 Memory Leakage
The memory detection detects memory leaks on the side of Device, which are usually caused by the developers' failure to release correctly the memory of applications for use of the AscendCL interface, and because the concept of memory distribution does not currently exist for the internal storage (Local Memory), the memory leak may only appear on GM. By specifying the command line parameter, --leak-check=yes, the memory leak detection can be activated.

Example report:
```text
====== ERROR : LeakCheck: detected memory leaks
// Memory leak detected
====== Direct leak of 100 byte(s)
// Each specific memory leak information
====== at 0x124080013000 on GM allocated in add_custom.cpp:14 (serialNo: 37)
====== Direct leak of 1000 byte(s)
====== at 0x124080014000 on GM allocated in add_custom.cpp:15 (serialNo: 55)
====== SUMMARY: 1100 byte(s) leaked in 2 allocation(s)
// Summary of all memory leakages, including the number of leaks and the total number of bytes leaked
```

In the example above, the first memory leak information contains address space, memory address, memory length and code location information, which points to the file name and line number where the memory is specifically assigned.
### 4.5 Illegal Releases
Illegal release refers to an operation to release an undistributed or released address, generally on GM.

Example report:
```text
====== ERROR: illegal free()
// Basic information of an unusual nature indicates that there was an irregularity of illegal release.
====== at 0x124080013000 on GM
// Anomalous memory location information, containing the address space and memory address that occurred
====== code in add_custom.cpp:84 (serialNo:63)
// Anomalous code locator information containing the serial number of the file name, line number and call api behavior
```
In the above example, the "0x12408013,000" address on GM was illegally released and the instructions that led to the anomaly corresponded to line 84 of the operator material add_custom.cpp.

### 4.6 Distribution memory unused
The distribution memory is unused, meaning that operator runtime applied for the memory, but did not use it until operator's operation was completed. The anomaly scenario is usually that operator used the wrong memory or there was a problem with operator's logic, usually on GM.

Example report:
```text
====== WARNING : Unused memory of 1000 byte(s)
// Basic information on anomalies, indicating that memory distribution was detected as not using an anomaly
====== at 1240c0016000 on GM
// Anomalous memory location information, containing the address space and memory address that occurred
====== code in add_custom.cpp:2 (serialNo: 69)
// Anomalous code locator information containing the serial number of the file name, line number and call api behavior
====== SUMMARY: 1100 byte(s) unused memory in 2 allocation(s)
// Memory Allocation Unused Summary Information, including Numbers of Unused Memory Blocks and Bytes
```