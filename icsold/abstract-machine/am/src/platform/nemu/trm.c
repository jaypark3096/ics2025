#include <am.h>
#include <nemu.h>

extern char _heap_start;
int main(const char* args);

// 堆区域定义，确保PMEM_END是正确的物理内存上限
Area heap = RANGE(&_heap_start, PMEM_END);
static const char mainargs[MAINARGS_MAX_LEN] = TOSTRING(MAINARGS_PLACEHOLDER); // 由CFLAGS定义

// 正确定义COM1串口的I/O端口地址（标准地址为0x3f8）
//#define SERIAL_PORT 0x3f8

void putch(char ch) {
	// 等待串口发送缓冲区为空（检查状态寄存器，偏移5）
	while ((inb(SERIAL_PORT + 5) & 0x20) == 0);
	// 向串口数据寄存器写入字符
	outb(SERIAL_PORT, ch);
}

void halt(int code) {
	nemu_trap(code);

	// 防止意外退出循环
	while (1);
}

void _trm_init() {
	int ret = main(mainargs);
	halt(ret);
}
