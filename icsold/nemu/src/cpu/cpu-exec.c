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

#include <cpu/cpu.h>
#include <cpu/decode.h>
#include <cpu/difftest.h>
#include <locale.h>
#include "../monitor/sdb/sdb.h"
/* The assembly code of instructions executed is only output to the screen
 * when the number of instructions executed is less than this value.
 * This is useful when you use the `si' command.
 * You can modify this value as you want.
 */

#define IRINGBUF_SIZE 16  // 环形缓冲区大小，可调整

 // 存储单条指令信息的结构体
typedef struct {
    vaddr_t pc;
    uint8_t inst[4];  // 假设最大指令长度为4字节
    int ilen;         // 指令实际长度
    char disasm[128]; // 反汇编字符串
} InstInfo;


#ifdef CONFIG_ITRACE
// 环形缓冲区全局变量
static InstInfo iringbuf[IRINGBUF_SIZE];
static int iringbuf_head = 0;  // 指向最新插入的位置
static int iringbuf_count = 0; // 当前缓冲区中的指令数量
#endif
#define MAX_INST_TO_PRINT 10

CPU_state cpu = {};
uint64_t g_nr_guest_inst = 0;
static uint64_t g_timer = 0; // unit: us
static bool g_print_step = false;

void device_update();

static void trace_and_difftest(Decode* _this, vaddr_t dnpc) {
#ifdef CONFIG_ITRACE_COND
    if (ITRACE_COND) { log_write("%s\n", _this->logbuf); }
#endif
    if (g_print_step) { IFDEF(CONFIG_ITRACE, puts(_this->logbuf)); }
    IFDEF(CONFIG_DIFFTEST, difftest_step(_this->pc, dnpc));

    // 执行监视点检查：若表达式值变化，暂停CPU执行
    if (check_wp()) {
        nemu_state.state = NEMU_STOP; // 将CPU状态设为停止
    }
}

#ifdef CONFIG_ITRACE
static void iringbuf_push(vaddr_t pc, uint8_t* inst, int ilen, const char* disasm) {
    // 复制指令信息到缓冲区
    iringbuf[iringbuf_head].pc = pc;
    memcpy(iringbuf[iringbuf_head].inst, inst, ilen);
    iringbuf[iringbuf_head].ilen = ilen;
    strncpy(iringbuf[iringbuf_head].disasm, disasm, sizeof(iringbuf[iringbuf_head].disasm) - 1);

    // 更新缓冲区指针
    iringbuf_head = (iringbuf_head + 1) % IRINGBUF_SIZE;
    if (iringbuf_count < IRINGBUF_SIZE) {
        iringbuf_count++;
    }
}
#endif

static void exec_once(Decode* s, vaddr_t pc) {
    s->pc = pc;
    s->snpc = pc;
    isa_exec_once(s);
    cpu.pc = s->dnpc;
#ifdef CONFIG_ITRACE
    char* p = s->logbuf;
    p += snprintf(p, sizeof(s->logbuf), FMT_WORD ":", s->pc);
    int ilen = s->snpc - s->pc;
    int i;
    uint8_t* inst = (uint8_t*)&s->isa.inst;
#ifdef CONFIG_ISA_x86
    for (i = 0; i < ilen; i++) {
#else
    for (i = ilen - 1; i >= 0; i--) {
#endif
        p += snprintf(p, 4, " %02x", inst[i]);
    }
    int ilen_max = MUXDEF(CONFIG_ISA_x86, 8, 4);
    int space_len = ilen_max - ilen;
    if (space_len < 0) space_len = 0;
    space_len = space_len * 3 + 1;
    memset(p, ' ', space_len);
    p += space_len;

    void disassemble(char* str, int size, uint64_t pc, uint8_t * code, int nbyte);
    disassemble(p, s->logbuf + sizeof(s->logbuf) - p,
        MUXDEF(CONFIG_ISA_x86, s->snpc, s->pc), (uint8_t*)&s->isa.inst, ilen);

    // 新增：将指令信息存入环形缓冲区
    iringbuf_push(s->pc, inst, ilen, p);
#endif
    }

static void execute(uint64_t n) {
    Decode s;
    for (;n > 0; n--) {
        exec_once(&s, cpu.pc);
        g_nr_guest_inst++;
        trace_and_difftest(&s, cpu.pc);
        if (nemu_state.state != NEMU_RUNNING) break;
        IFDEF(CONFIG_DEVICE, device_update());
    }
}

static void statistic() {
    IFNDEF(CONFIG_TARGET_AM, setlocale(LC_NUMERIC, ""));
#define NUMBERIC_FMT MUXDEF(CONFIG_TARGET_AM, "%", "%'") PRIu64
    Log("host time spent = " NUMBERIC_FMT " us", g_timer);
    Log("total guest instructions = " NUMBERIC_FMT, g_nr_guest_inst);
    if (g_timer > 0) Log("simulation frequency = " NUMBERIC_FMT " inst/s", g_nr_guest_inst * 1000000 / g_timer);
    else Log("Finish running in less than 1 us and can not calculate the simulation frequency");
}

void assert_fail_msg() {
    isa_reg_display();
    statistic();
}

/* Simulate how the CPU works. */
void cpu_exec(uint64_t n) {
    g_print_step = (n < MAX_INST_TO_PRINT);
    switch (nemu_state.state) {
    case NEMU_END: case NEMU_ABORT: case NEMU_QUIT:
        printf("Program execution has ended. To restart the program, exit NEMU and run again.\n");
        return;
    default: nemu_state.state = NEMU_RUNNING;
    }

    uint64_t timer_start = get_time();

    execute(n);

    uint64_t timer_end = get_time();
    g_timer += timer_end - timer_start;

    switch (nemu_state.state) {
    case NEMU_RUNNING: nemu_state.state = NEMU_STOP; break;

    case NEMU_END: case NEMU_ABORT:
        Log("nemu: %s at pc = " FMT_WORD,
            (nemu_state.state == NEMU_ABORT ? ANSI_FMT("ABORT", ANSI_FG_RED) :
                (nemu_state.halt_ret == 0 ? ANSI_FMT("HIT GOOD TRAP", ANSI_FG_GREEN) :
                    ANSI_FMT("HIT BAD TRAP", ANSI_FG_RED))),
            nemu_state.halt_pc);
        // fall through
    case NEMU_QUIT: statistic();
    }
}

// 在cpu-exec.c中添加
void print_iringbuf(vaddr_t err_pc) {
#ifdef CONFIG_ITRACE
    printf("Last %d instructions before error:\n", iringbuf_count);

    // 计算起始位置
    int start = (iringbuf_head - iringbuf_count + IRINGBUF_SIZE) % IRINGBUF_SIZE;

    for (int i = 0; i < iringbuf_count; i++) {
        int idx = (start + i) % IRINGBUF_SIZE;
        InstInfo* info = &iringbuf[idx];

        // 打印PC
        printf("  %s" FMT_WORD ": ",
            (info->pc == err_pc) ? "-->" : "  ",  // 标记错误指令
            info->pc);

        // 打印指令二进制
        for (int j = 0; j < info->ilen; j++) {
            printf("%02x ", info->inst[j]);
        }

        // 补齐空格使格式对齐
        for (int j = info->ilen; j < 4; j++) {
            printf("   ");
        }

        // 打印反汇编
        printf("%s\n", info->disasm);
    }
#else
    printf("Instruction trace is not enabled (CONFIG_ITRACE not defined)\n");
#endif
}
