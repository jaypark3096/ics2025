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

#include <isa.h>
#include <cpu/difftest.h>

// 定义mstatus寄存器中的中断使能位（MIE）
#define MSTATUS_MIE (1 << 3)

word_t isa_raise_intr(word_t NO, vaddr_t epc) {
	/* 模拟RISC-V异常响应机制：
	 * 1. 设置mepc为异常发生的下一条指令地址（epc）
	 * 2. 设置mcause为异常编号（NO）
	 * 3. 返回mtvec寄存器中存储的异常入口地址
	 */
	cpu.mepc = epc;       // 保存异常发生位置
	cpu.mcause = NO;      // 记录异常原因
	return cpu.mtvec;     // 返回异常入口地址（由cte_init设置）
}

word_t isa_query_intr() {
	return INTR_EMPTY;
}
