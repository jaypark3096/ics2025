// cte.c
#include <am.h>
#include <riscv/riscv.h>
#include <klib.h>
#include <stdint.h>

static Context* (*user_handler)(Event, Context*) = NULL;

Context* __am_irq_handle(Context* c) {
    if (user_handler) {
        Event ev = { 0 };

        // 根据mcause识别事件类型
        uintptr_t mcause = c->cause;

        // 检查最高位，判断是中断还是异常
        if (mcause & (1UL << 31)) {
            // 这是中断
            uintptr_t interrupt_code = mcause & 0x7FFFFFFF;
            switch (interrupt_code) {
            case 0: // 软件中断
            case 1: // 时钟中断
            case 2: // 外部中断
                // 暂时不处理中断
                ev.event = EVENT_ERROR;
                break;
            default:
                ev.event = EVENT_ERROR;
                break;
            }
        }
        else {
            // 这是异常
            uintptr_t exception_code = mcause;
            switch (exception_code) {
            case 8:  // 用户模式ecall
            case 11: // 机器模式ecall  
                ev.event = EVENT_YIELD;
                // 对于ecall异常，需要将epc+4指向下一条指令
                c->epc += 4;
                break;
            default:
                ev.event = EVENT_ERROR;
                ev.cause = mcause;
                break;
            }
        }

        c = user_handler(ev, c);
        assert(c != NULL);
    }
    return c;
}

extern void __am_asm_trap(void);

bool cte_init(Context* (*handler)(Event, Context*)) {
    // 设置异常入口地址
    asm volatile("csrw mtvec, %0" : : "r"(__am_asm_trap));

    // 注册事件处理回调
    user_handler = handler;

    return true;
}

void yield() {
    // 使用ecall指令触发自陷
    asm volatile("ecall");
}

Context* kcontext(Area kstack, void (*entry)(void*), void* arg) {
    return NULL;
}

bool ienabled() {
    return false;
}

void iset(bool enable) {
    // 暂时不实现中断使能
}