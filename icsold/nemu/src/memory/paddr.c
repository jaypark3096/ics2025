/***************************************************************************************
* Copyright (c) 2014-2024 Zihao Yu, Nanjing University
*
* NEMU is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
*
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
*
* See the Mulan PSL v2 for more details.
***************************************************************************************/

#include <memory/host.h>
#include <memory/paddr.h>
#include <device/mmio.h>
#include <isa.h>
#include <cpu/cpu.h>  
#include <utils.h>  

// 将 mtrace_log 函数声明移到所有条件编译块之前
#ifdef CONFIG_MTRACE
static void mtrace_log(bool is_write, paddr_t addr, int len, word_t data);
#else
// 未定义 CONFIG_MTRACE 时，提供空实现以避免编译错误
static void mtrace_log(bool is_write, paddr_t addr, int len, word_t data) {
    // 空函数，什么也不做
}
#endif

#if   defined(CONFIG_PMEM_MALLOC)
static uint8_t* pmem = NULL;
#else // CONFIG_PMEM_GARRAY
static uint8_t pmem[CONFIG_MSIZE] PG_ALIGN = {};
#endif

uint8_t* guest_to_host(paddr_t paddr) { return pmem + paddr - CONFIG_MBASE; }
paddr_t host_to_guest(uint8_t* haddr) { return haddr - pmem + CONFIG_MBASE; }

static word_t pmem_read(paddr_t addr, int len) {
    word_t ret = host_read(guest_to_host(addr), len);
    return ret;
}

static void pmem_write(paddr_t addr, int len, word_t data) {
    host_write(guest_to_host(addr), len, data);
}

/*static void out_of_bound(paddr_t addr) {
    print_iringbuf(cpu.pc);

    panic("address = " FMT_PADDR " is out of bound of pmem [" FMT_PADDR ", " FMT_PADDR "] at pc = " FMT_WORD,
        addr, PMEM_LEFT, PMEM_RIGHT, cpu.pc);
}*/

// 定义设备内存映射区域
typedef struct {
    paddr_t start;
    paddr_t end;
    const char* name;
} iomap_t;

// 常见的设备内存映射区域
static iomap_t iomap[] = {
    {0xa0000000, 0xa0000fff, "VGA/Display"},      // 显示设备区域
    {0xa0000000, 0xa0000007, "RTC"},            // 实时时钟
    {0xa0000048, 0xa000004b, "Serial"},         // 串口
    {0xa0000060, 0xa0000063, "i8042"},          // 键盘控制器
    {0xa0000068, 0xa000006b, "i8042"},          // 键盘控制器状态
    {0xa0000100, 0xa0000107, "IDE"},            // IDE控制器
    {0xa0000200, 0xa0000203, "Audio"},          // 音频设备
    // 可以根据需要添加更多设备区域
};

#define IO_MAP_SIZE (sizeof(iomap) / sizeof(iomap[0]))

// 检查地址是否在设备内存映射区域内
static bool in_iomap(paddr_t addr) {
    for (size_t i = 0; i < IO_MAP_SIZE; i++) {
        if (addr >= iomap[i].start && addr <= iomap[i].end) {
            return true;
        }
    }
    return false;
}

// mtrace_log 函数定义移到条件编译块之外
#ifdef CONFIG_MTRACE
static void mtrace_log(bool is_write, paddr_t addr, int len, word_t data) {
#ifdef CONFIG_MTRACE_COND
    if (!(CONFIG_MTRACE_COND)) return;
#endif

    const char* op = is_write ? "W" : "R";
    Log("%s: " FMT_PADDR " (len=%d) data=0x%08x", op, addr, len, data);
}
#endif

void init_mem() {
#if   defined(CONFIG_PMEM_MALLOC)
    pmem = malloc(CONFIG_MSIZE);
    assert(pmem);
#endif
#ifdef CONFIG_MEM_RANDOM
    memset(pmem, rand(), CONFIG_MSIZE);
#endif
    Log("physical memory area [" FMT_PADDR ", " FMT_PADDR "]", PMEM_LEFT, PMEM_RIGHT);

    // 记录设备内存映射区域信息
    for (size_t i = 0; i < IO_MAP_SIZE; i++) {
        Log("IO map: %s [" FMT_PADDR ", " FMT_PADDR "]",
            iomap[i].name, iomap[i].start, iomap[i].end);
    }
}

word_t paddr_read(paddr_t addr, int len) {
    word_t data;

    // 首先检查是否为设备内存映射区域
    if (in_iomap(addr)) {
#ifdef CONFIG_DEVICE
        data = mmio_read(addr, len);
#else
        out_of_bound(addr);
        data = 0;
#endif
    }
    else if (likely(in_pmem(addr))) {
        data = pmem_read(addr, len);
    }
    else {
#ifdef CONFIG_DEVICE
        data = mmio_read(addr, len);
#else
        out_of_bound(addr);
        data = 0;
#endif
    }

    mtrace_log(false, addr, len, data);
    return data;
}

void paddr_write(paddr_t addr, int len, word_t data) {
    // 首先检查是否为设备内存映射区域
    if (in_iomap(addr)) {
#ifdef CONFIG_DEVICE
        mmio_write(addr, len, data);
#else
        out_of_bound(addr);
#endif
    }
    else if (likely(in_pmem(addr))) {
        pmem_write(addr, len, data);
    }
    else {
#ifdef CONFIG_DEVICE
        mmio_write(addr, len, data);
#else
        out_of_bound(addr);
#endif
    }

    mtrace_log(true, addr, len, data);
}
