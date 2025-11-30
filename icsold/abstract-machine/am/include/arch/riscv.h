// riscv.h
#ifndef __ARCH_RISCV_H__
#define __ARCH_RISCV_H__

#include <stdint.h>

typedef struct Context {
    // 通用寄存器（按trap.S中PUSH的顺序）
    uintptr_t x1;   // ra
    uintptr_t x3;   // gp
    uintptr_t x4;   // tp
    uintptr_t x5;   // t0
    uintptr_t x6;   // t1
    uintptr_t x7;   // t2
    uintptr_t x8;   // s0/fp
    uintptr_t x9;   // s1
    uintptr_t x10;  // a0
    uintptr_t x11;  // a1
    uintptr_t x12;  // a2
    uintptr_t x13;  // a3
    uintptr_t x14;  // a4
    uintptr_t x15;  // a5
    uintptr_t x16;  // a6
    uintptr_t x17;  // a7
    uintptr_t x18;  // s2
    uintptr_t x19;  // s3
    uintptr_t x20;  // s4
    uintptr_t x21;  // s5
    uintptr_t x22;  // s6
    uintptr_t x23;  // s7
    uintptr_t x24;  // s8
    uintptr_t x25;  // s9
    uintptr_t x26;  // s10
    uintptr_t x27;  // s11
    uintptr_t x28;  // t3
    uintptr_t x29;  // t4
    uintptr_t x30;  // t5
    uintptr_t x31;  // t6

    // 异常相关特殊信息（按trap.S中存储顺序）
    uintptr_t cause;   // 对应偏移 NR_REGS * XLEN
    uintptr_t status;  // 对应偏移 (NR_REGS + 1) * XLEN
    uintptr_t epc;     // 对应偏移 (NR_REGS + 2) * XLEN

    // 地址空间信息
    void* pdir;
} Context;

#endif